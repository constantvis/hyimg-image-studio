"""The segmentation worker (owner 2026-10-10: «давай качай конечно»): SAM 3 (MLX, 4-bit) finds objects by a word, a click or a box,
BiRefNet lite draws their edges. It runs in its own Python, ~/Library/Caches/Hyimg/venvs/seg (mlx, mlx-vlm, torch), never in the
board's server, which talks to it through seg.py: one JSON request a line on stdin, one JSON answer a line on stdout, the pictures as
PNG files in a folder the server gives. The models load on their first use and stay; the backbone of the last pictures is kept, so a
second word or click on the same picture skips it. The worker leaves when stdin closes (the server is gone) or after HYIMG_SEG_IDLE
seconds without a request (600 by default: the models hold about 2 GB).

  {"op": "segment", "image": PNG, "out": DIR, "text"?: "phone", "points"?: [[x, y, 1|0], …], "box"?: [x, y, w, h]}
      → {"masks": [PNG, …], "boxes": [[x, y, w, h], …], "scores": [...], "ms", "how": "text"|"points"|"box"}
        a word: every instance SAM 3 finds over 0.3, best first (with a box or points: only those over the box or under a positive point);
        points or a box alone: the one object under them (SAM's interactive decoder, label 1 the object, 0 not)
  {"op": "matte", "image": PNG, "out": DIR, "mask"?: PNG, "box"?: [x, y, w, h], "band"?: px}
      → {"alpha": PNG, "ms"}: BiRefNet lite on a crop around the mask or the box (the whole picture without either), its alpha back at
        full size; with a mask, the mask's edge refined (edges.refine: guided by the picture, cut where BiRefNet sees background)
  {"op": "ping"} → {"ok": true, "loaded": [...], "stub": bool};  {"op": "warm", "models": ["sam", "bir"]} loads them before they are needed

HYIMG_SEG_STUB=1 answers without models (tests): a word gives a centred ellipse, points a disc, a box itself; matte blurs the mask.
"""
import importlib.util
import json
import os
import select
import sys
import time

import numpy as np
from PIL import Image, ImageFilter

sys.dont_write_bytecode = True   # nothing written into the models' folders (BiRefNet's code is imported from there)
MODELS = os.path.expanduser(os.environ.get("HYIMG_MODELS", "~/Library/Caches/Hyimg/models"))
SAM_DIR, BIR_DIR = os.path.join(MODELS, "sam3-4bit"), os.path.join(MODELS, "birefnet-lite")
STUB = os.environ.get("HYIMG_SEG_STUB") == "1"
IDLE = float(os.environ.get("HYIMG_SEG_IDLE", "600"))
THRESHOLD = 0.3
_spec = importlib.util.spec_from_file_location("hyimg_seg_edges", os.path.join(os.path.dirname(os.path.abspath(__file__)), "edges.py"))
EDGES = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(EDGES)   # by its path: python -I puts no folder of ours on sys.path


def _save(a, path):
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(path)
    return path


def _xywh(b):
    x0, y0, x1, y1 = (float(v) for v in b)
    return [round(x0), round(y0), round(x1 - x0), round(y1 - y0)]


def _soft(logits, size):
    """a mask's logits (h × w) at the picture's size: bilinear, cut at 0, one pixel of antialiasing"""
    W, H = size
    m = np.asarray(Image.fromarray(logits.astype(np.float32)).resize((W, H), Image.BILINEAR)) > 0
    return np.asarray(Image.fromarray(m.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(0.7)))


def _keep_near(mask, pts):
    """a click's mask without the specks SAM leaves around the picture: the parts under a positive point, or as big as a tenth of
    the largest part next to them, stay"""
    from scipy import ndimage
    on = mask > 127
    lab, n = ndimage.label(on)
    if n < 2:
        return mask
    sizes = ndimage.sum(on, lab, range(1, n + 1))
    keep = {int(lab[int(y), int(x)]) for x, y, l in pts if l == 1 and 0 <= int(y) < lab.shape[0] and 0 <= int(x) < lab.shape[1]} - {0}
    big = sizes.max()
    keep |= {i + 1 for i, s in enumerate(sizes) if s >= big * 0.1 and (not keep or s >= big * 0.25)}
    if not keep:
        return mask
    out = mask.copy()
    out[~np.isin(lab, list(keep)) & on] = 0
    return out


class Sam:
    def __init__(self):
        import mlx.core as mx
        from mlx_vlm.utils import load_model
        from mlx_vlm.models.sam3.generate import Sam3Predictor
        from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
        from pathlib import Path
        self.mx = mx
        self.model = load_model(Path(SAM_DIR))
        self.proc = Sam3Processor.from_pretrained(SAM_DIR)
        self.pred = Sam3Predictor(self.model, self.proc, score_threshold=THRESHOLD)
        self.cache = []   # [(key, backbone, tracker fpn)], the last 2 pictures

    def feats(self, key, im):
        for k, b, f in self.cache:
            if k == key:
                return b, f
        mx = self.mx
        from mlx_vlm.models.sam3.generate import _get_backbone_features
        b = _get_backbone_features(self.model, mx.array(self.proc.preprocess_image(im)["pixel_values"]))
        f = self.model.tracker_neck(b); mx.eval(f)
        self.cache = [(key, b, f)] + self.cache[:1]
        return b, f

    def text(self, im, key, text):
        """mlx_vlm's _detect_with_backbone, with the masks' logits kept for the edge (it gives them cut at 0)"""
        mx = self.mx
        from mlx_vlm.models.sam3.generate import _get_det_features, _nms
        bb, _ = self.feats(key, im)
        det = self.model.detector_model
        src, pos, fpn, (hf, wf) = _get_det_features(self.model, bb)
        emb, am = self.pred._get_input_embeddings(text)
        enc = det.detr_encoder(src, pos, emb, am)
        hs, ref, pres = det.detr_decoder(vision_features=enc, inputs_embeds=emb, vision_pos_encoding=pos, text_mask=am, spatial_shape=(hf, wf))
        logits = det.dot_product_scoring(hs, emb, am)[-1].squeeze(-1)
        seg = det.mask_decoder(hs[-1], list(fpn), encoder_hidden_states=enc, prompt_features=emb, prompt_mask=am)
        sc = mx.sigmoid(logits[0].squeeze())
        if pres[-1] is not None:
            sc = sc * mx.sigmoid(pres[-1][0])
        mx.eval(sc, ref, seg)
        sc = np.atleast_1d(np.array(sc)); boxes = np.array(ref[-1][0]); masks = np.array(seg["pred_masks"][0])
        keep = np.nonzero(sc > THRESHOLD)[0]
        if not len(keep):
            return [], [], []
        keep = keep[_nms(boxes[keep], sc[keep])]
        W, H = im.size
        out = []
        for i in keep[np.argsort(-sc[keep])]:
            cx, cy, w, h = boxes[i]
            out.append((_soft(masks[i], im.size), [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], float(sc[i])))
        return [o[0] for o in out], [o[1] for o in out], [o[2] for o in out]

    def interactive(self, im, key, pts, box):
        """one object under points (label 1 is it, 0 is not) and/or a box: SAM's prompt encoder and mask decoder on the tracker's
        features, the three answers of multimask, the one SAM rates best"""
        mx = self.mx
        _, fpn = self.feats(key, im)
        tm = self.model.tracker_model
        W, H = im.size
        B, h, w, D = fpn[2].shape
        g = tm.prompt_encoder.image_embedding_size   # the points in the embedding's grid (sam_components._embed_points)
        points = None
        if pts:
            points = (mx.array(np.array([[[x * g[1] / W, y * g[0] / H] for x, y, _ in pts]], np.float32)),
                      mx.array(np.array([[1 if l == 1 else 0 for _, _, l in pts]], np.int32)))
        boxes = mx.array(np.array([[box[0] / W, box[1] / H, (box[0] + box[2]) / W, (box[1] + box[3]) / H]], np.float32)) if box else None
        src = fpn[2].reshape(B, h * w, D) + tm.no_memory_embedding
        pe = mx.broadcast_to(tm.prompt_encoder.get_dense_pe(), (B, h * w, D))
        sp, de = tm.prompt_encoder(points=points, boxes=boxes, masks=None)
        m, iou, _, _ = tm.mask_decoder(image_embeddings=src, image_pe=pe, sparse_prompt_embeddings=sp, dense_prompt_embeddings=de,
                                       multimask_output=True, high_res_features=[fpn[1], fpn[0]])
        mx.eval(m, iou)
        m, iou = np.array(m[0]), np.array(iou[0])
        best = int(np.argmax(iou))
        mask = _soft(m[best], im.size)
        if pts:
            mask = _keep_near(mask, pts)
        if box:   # a box keeps the object inside it (a little past its edges): SAM's specks around the picture go
            x, y, bw, bh = box; p = 0.05 * max(bw, bh); keep = np.zeros_like(mask)
            keep[max(0, int(y - p)):int(y + bh + p), max(0, int(x - p)):int(x + bw + p)] = 1; mask = mask * keep
        ys, xs = np.nonzero(mask > 127)
        bx = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1] if len(xs) else [0, 0, 0, 0]
        return [mask], [bx], [float(1 / (1 + np.exp(-iou[best])))]


class Bir:
    SIZE = 1024

    def __init__(self):
        import torch
        from safetensors.torch import load_file
        self.torch = torch
        # the model's own code (birefnet.py, MIT, read before it was first run here) as a package of its folder: it imports its config
        # relatively; its backbone weights are not fetched (bb_pretrained false), the weights are model.safetensors
        spec = importlib.util.spec_from_file_location("hyimg_birefnet", os.path.join(BIR_DIR, "__init__.py"), submodule_search_locations=[BIR_DIR])
        pkg = importlib.util.module_from_spec(spec); sys.modules["hyimg_birefnet"] = pkg
        cfg = importlib.import_module("hyimg_birefnet.BiRefNet_config").BiRefNetConfig(bb_pretrained=False)
        net = importlib.import_module("hyimg_birefnet.birefnet").BiRefNet(config=cfg)
        net.load_state_dict(load_file(os.path.join(BIR_DIR, "model.safetensors")), strict=False)
        self.dev = "mps" if torch.backends.mps.is_available() and os.environ.get("HYIMG_SEG_CPU") != "1" else "cpu"
        self.net = net.eval().to(self.dev)

    def alpha(self, rgb):
        """H × W × 3 uint8 → H × W float 0..1, the foreground BiRefNet sees"""
        torch = self.torch
        h, w = rgb.shape[:2]
        x = np.asarray(Image.fromarray(rgb).resize((self.SIZE, self.SIZE), Image.BILINEAR), np.float32) / 255
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        t = torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32)).to(self.dev)
        with torch.no_grad():
            p = self.net(t)[-1].sigmoid()[0, 0].float().cpu().numpy()
        return np.asarray(Image.fromarray(p).resize((w, h), Image.BILINEAR)).clip(0, 1)


def _crop_box(b, W, H, pad=0.15):
    x, y, w, h = b
    p = max(16, int(max(w, h) * pad))
    return max(0, int(x) - p), max(0, int(y) - p), min(W, int(x + w) + p), min(H, int(y + h) + p)


def matte(bir, rgb, mask=None, box=None, band=None):
    H, W = rgb.shape[:2]
    if mask is not None and box is None:
        ys, xs = np.nonzero(mask > 127)
        if not len(xs):
            return np.zeros((H, W), np.uint8)
        box = [xs.min(), ys.min(), xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min()]
    x0, y0, x1, y1 = _crop_box(box, W, H) if box is not None else (0, 0, W, H)
    a = np.zeros((H, W), np.float32)
    a[y0:y1, x0:x1] = bir.alpha(np.ascontiguousarray(rgb[y0:y1, x0:x1])) if bir else 0
    if mask is not None:   # the mask's edge onto the picture's, cut where BiRefNet sees background (edges.py)
        return EDGES.refine(rgb, mask, band, np.rint(a * 255).astype(np.uint8))
    elif box is not None:
        keep = np.zeros_like(a); keep[y0:y1, x0:x1] = 1; a *= keep
    return np.rint(a * 255).astype(np.uint8)


class Stub:
    """no models: shapes enough for the routes and the studio's tools to be tested quickly"""
    def segment(self, im, req):
        W, H = im.size
        yy, xx = np.mgrid[0:H, 0:W]
        if req.get("text"):
            m = ((xx - W / 2) / (W / 4)) ** 2 + ((yy - H / 2) / (H / 4)) ** 2 <= 1
        elif req.get("points"):
            x, y = req["points"][0][:2]; r = max(8, min(W, H) / 8)
            m = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        else:
            x, y, w, h = req["box"]; m = (xx >= x) & (xx < x + w) & (yy >= y) & (yy < y + h)
        ys, xs = np.nonzero(m)
        return [m.astype(np.uint8) * 255], [[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], [0.9]


def handle(req, st):
    op = req.get("op")
    if op == "ping":
        return {"ok": True, "loaded": [k for k in ("sam", "bir") if st.get(k)], "stub": STUB}
    if op == "warm":   # load the models before the first click (the studio asks when Smart Select is picked); first runs compile
        t0 = time.time()
        if not STUB:
            for k in req.get("models") or ["sam"]:
                if k == "sam" and not st.get("sam"): st["sam"] = Sam()
                if k == "bir" and not st.get("bir"): st["bir"] = Bir(); st["bir"].alpha(np.zeros((64, 64, 3), np.uint8))
        return {"ok": True, "loaded": [k for k in ("sam", "bir") if st.get(k)], "ms": round((time.time() - t0) * 1000)}
    t0 = time.time()
    im = Image.open(req["image"]).convert("RGB")
    out = req["out"]
    if op == "segment":
        text, pts, box = (req.get("text") or "").strip(), req.get("points") or [], req.get("box")
        pts = [(float(p[0]), float(p[1]), 1 if len(p) < 3 or float(p[2]) > 0 else 0) for p in pts]
        if STUB:
            masks, boxes, scores = Stub().segment(im, req); how = "text" if text else "points" if pts else "box"
        else:
            if not st.get("sam"):
                t = time.time(); st["sam"] = Sam(); st["load_ms"] = round((time.time() - t) * 1000)
            sam, key = st["sam"], req.get("key") or req["image"]
            if text:
                masks, boxes, scores = sam.text(im, key, text); how = "text"
                if box or pts:   # a word with a box or points: only the instances over the box or under a positive point
                    sel = []
                    for i, (m, b) in enumerate(zip(masks, boxes)):
                        over = box and min(b[2], box[0] + box[2]) > max(b[0], box[0]) and min(b[3], box[1] + box[3]) > max(b[1], box[1])
                        under = any(l == 1 and 0 <= int(y) < m.shape[0] and 0 <= int(x) < m.shape[1] and m[int(y), int(x)] > 127 for x, y, l in pts)
                        if over or under: sel.append(i)
                    masks, boxes, scores = [masks[i] for i in sel], [boxes[i] for i in sel], [scores[i] for i in sel]
            elif pts or box:
                masks, boxes, scores = sam.interactive(im, key, pts, box); how = "points" if pts else "box"
            else:
                return {"error": "segment: text, points or a box"}
        files = [_save(m, os.path.join(out, f"mask-{i}.png")) for i, m in enumerate(masks)]
        return {"masks": files, "boxes": [_xywh(b) for b in boxes], "scores": [round(s, 4) for s in scores], "how": how,
                "ms": round((time.time() - t0) * 1000), "load_ms": st.pop("load_ms", None)}
    if op == "matte":
        rgb = np.asarray(im)
        mask = np.asarray(Image.open(req["mask"]).convert("L").resize(im.size)) if req.get("mask") else None
        if STUB:
            base = mask if mask is not None else np.full(rgb.shape[:2], 255, np.uint8)
            a = np.asarray(Image.fromarray(base).filter(ImageFilter.GaussianBlur(2)))
        else:
            if not st.get("bir"):
                t = time.time(); st["bir"] = Bir(); st["load_ms"] = round((time.time() - t) * 1000)
            a = matte(st["bir"], rgb, mask, req.get("box"), req.get("band"))
        return {"alpha": _save(a, os.path.join(out, "alpha.png")), "ms": round((time.time() - t0) * 1000), "load_ms": st.pop("load_ms", None),
                "device": None if STUB else st["bir"].dev}
    return {"error": f"unknown op {op!r}"}


def main():
    st = {}
    out = sys.stdout
    sys.stdout = sys.stderr   # what the libraries print goes to the log, the answers alone to stdout
    while True:
        r, _, _ = select.select([sys.stdin], [], [], IDLE if IDLE > 0 else None)
        if not r:
            break   # idle: the models give their memory back; the server starts the worker again on the next request
        line = sys.stdin.readline()
        if not line:
            break
        try:
            res = handle(json.loads(line), st)
        except Exception as ex:   # one bad request answers with its error, the worker stays
            import traceback
            traceback.print_exc()
            res = {"error": f"{type(ex).__name__}: {ex}"}
        out.write(json.dumps(res) + "\n"); out.flush()


if __name__ == "__main__":
    main()
