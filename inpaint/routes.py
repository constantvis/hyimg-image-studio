"""The frames plugin's server routes (owner 2026-10-05: the editor's content-aware fill and subject mask live in Hyimg's own server, not in
a dev server beside it). Hyimg loads this file by manifest.json "server" and serves each route at POST /api/plugin/<plugin>/<name>:

  POST .../inpaint {"image": "data:image/png;base64,…", "mask": "data:image/png;base64,…"} → {"image": "data:image/png;base64,…", "ms": 2900,
        "seconds": 2.9, "compute": "local · LaMa ONNX · CPU"}
        LaMa (lama.py) on this Mac; without the model file: 503 {"error", "fallback": "local"}, and the editor fills by itself
  POST .../subject {"image": "data:…"} → {"mask": "data:image/png;base64,…" (white = the subject), "ms": 500}, macOS Vision (subject.py)
  POST .../versions {"dir": "frames/<stamp>"} → {"versions": [0, 1], "next": 2}: the saved versions of a frame
  POST .../prune {"dir": "frames/<stamp>", "keep": 10} → {"kept", "removed"}: older versions go, with the files only they used
  POST .../segment {"image", "text"?, "points"?: [{x, y, label: 1|-1}], "box"?: {x, y, w, h}} → {"masks": [data URL], "boxes": [[x, y, w, h]],
        "scores", "how", "ms"}: SAM 3 (seg.py, segworker.py); a word gives every instance, best first; points or a box the one object
  POST .../matte {"image", "mask"? | "box"?, "method"?: "birefnet"|"guided"} → {"alpha": data URL, "how", "ms"}: BiRefNet lite on a crop
        around the mask or box (the whole picture without either: Remove Background); a mask's edge snaps to the picture's (edges.py).
        "guided" (or BiRefNet missing, with a mask) runs here, no model. Missing models: 503 {"error", "missing", "fallback"}
  POST .../segwarm {"models": ["sam", "bir"]} → loads them before the first click;  POST .../models {} → what Settings › Image Studio shows;
  POST .../models_remove {"id"} → that model's folder removed

A route is fn(body bytes, query dict) -> (status, content type, bytes). lama.py and subject.py are loaded under names of their own, so
nothing of this folder goes into the server's sys.path; the LaMa session stays loaded in the server's process after the first fill.
"""
import base64
import importlib.util
import io
import json
import os
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
_MODS, _LOCK = {}, threading.Lock()


def _mod(name):
    with _LOCK:
        if name not in _MODS:
            spec = importlib.util.spec_from_file_location(f"hyimg_frames_{name}", os.path.join(HERE, name + ".py"))
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); _MODS[name] = m
        return _MODS[name]


def _json(status, obj):
    return status, "application/json", json.dumps(obj, ensure_ascii=False).encode()


def _img(url, mode):
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[-1]))).convert(mode)


def _png_url(im):
    buf = io.BytesIO(); im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def inpaint(body, query=None):
    import numpy as np
    from PIL import Image
    lama = _mod("lama")
    if not os.path.exists(lama.MODEL):
        return _json(503, {"error": f"The LaMa model is missing: {lama.MODEL}", "fallback": "local"})
    try:
        d = json.loads(body); img = _img(d["image"], "RGBA"); mask = _img(d["mask"], "L").resize(img.size)
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as ex:   # AttributeError: an image that is not a string (unit test, 2026-10-06)
        return _json(400, {"error": f"bad request: {ex}"})
    t = time.time()
    out = lama.fill(np.asarray(img)[..., :3], np.asarray(mask))
    ms = round((time.time() - t) * 1000)
    res = Image.fromarray(out).convert("RGBA"); res.putalpha(img.getchannel("A"))
    # how long and where, in Hyimg's words for a file's json (owner 2026-10-11): the editor keeps them on the layer it makes (run)
    return _json(200, {"image": _png_url(res), "ms": ms, "seconds": round(ms / 1000, 3), "compute": "local · LaMa ONNX · CPU"})


def _seg_img(d, key="image"):
    """a data URL to a PNG file for the worker, and its hash (the worker keeps the backbone of the last pictures by it)"""
    import hashlib
    import tempfile
    raw = base64.b64decode(str(d[key]).split(",", 1)[-1])
    tmp = tempfile.mkdtemp(prefix="hyimg-seg-")
    path = os.path.join(tmp, f"{key}.png")
    with open(path, "wb") as f: f.write(raw)
    return tmp, path, hashlib.sha1(raw).hexdigest()


def _file_url(path):
    with open(path, "rb") as f: return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _xywh(b):
    if b is None: return None
    if isinstance(b, dict): return [float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])]
    return [float(v) for v in b][:4]


def segment(body, query=None):
    import shutil
    seg = _mod("seg")
    try:
        d = json.loads(body); text = str(d.get("text") or "").strip()
        pts = [[float(p["x"]), float(p["y"]), 1 if float(p.get("label", 1)) > 0 else 0] if isinstance(p, dict) else [float(p[0]), float(p[1]), 1 if len(p) < 3 or float(p[2]) > 0 else 0]
               for p in d.get("points") or []]
        box = _xywh(d.get("box"))
        if not (text or pts or box): raise ValueError("text, points or box")
        tmp, path, key = _seg_img(d)
    except (ValueError, KeyError, TypeError, AttributeError, IndexError) as ex:
        return _json(400, {"error": f"bad request: {ex}"})
    try:
        miss = seg.missing("sam3-4bit")
        if miss: return _json(503, miss)
        try: r = seg.call({"op": "segment", "image": path, "out": tmp, "key": key, "text": text, "points": pts, "box": box})
        except RuntimeError as ex: return _json(500, {"error": str(ex)})
        return _json(200, {"masks": [_file_url(m) for m in r["masks"]], "boxes": r["boxes"], "scores": r["scores"], "how": r["how"],
                           "ms": r["ms"], "loadMs": r.get("load_ms"), "model": "sam3-4bit"})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def matte(body, query=None):
    import shutil
    seg = _mod("seg")
    try:
        d = json.loads(body); method = d.get("method") or "birefnet"; box = _xywh(d.get("box"))
        if method not in ("birefnet", "guided"): raise ValueError("method: birefnet or guided")
        tmp, path, key = _seg_img(d)
        mpath = _seg_img(d, "mask")[1] if d.get("mask") else None
    except (ValueError, KeyError, TypeError, AttributeError) as ex:
        return _json(400, {"error": f"bad request: {ex}"})
    try:
        miss = seg.missing("birefnet-lite") if method == "birefnet" else None
        if miss and not mpath:   # Remove Background without BiRefNet: the studio asks macOS Vision (/subject) instead
            return _json(503, {**miss, "fallback": "vision"})
        if method == "guided" or miss:   # the edge without a model: the guided filter here
            import numpy as np
            from PIL import Image
            t = time.time(); img = _img(d["image"], "RGB"); mask = _img(d["mask"], "L").resize(img.size)
            a = _mod("edges").guided(np.asarray(img), np.asarray(mask))
            return _json(200, {"alpha": _png_url(Image.fromarray(a)), "how": "guided", "ms": round((time.time() - t) * 1000),
                               "note": miss and miss["error"]})
        try: r = seg.call({"op": "matte", "image": path, "out": tmp, "mask": mpath, "box": box})
        except RuntimeError as ex: return _json(500, {"error": str(ex)})
        return _json(200, {"alpha": _file_url(r["alpha"]), "how": "birefnet", "ms": r["ms"], "loadMs": r.get("load_ms"), "device": r.get("device")})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if mpath: shutil.rmtree(os.path.dirname(mpath), ignore_errors=True)


def segwarm(body, query=None):
    seg = _mod("seg")
    try: want = [m for m in (json.loads(body or b"{}").get("models") or ["sam"]) if m in ("sam", "bir")]
    except (ValueError, AttributeError) as ex: return _json(400, {"error": str(ex)})
    miss = seg.missing(*[{"sam": "sam3-4bit", "bir": "birefnet-lite"}[m] for m in want])
    if miss: return _json(503, miss)
    try: return _json(200, seg.call({"op": "warm", "models": want}))
    except RuntimeError as ex: return _json(500, {"error": str(ex)})


def models(body, query=None):
    return _json(200, _mod("seg").status())


def models_remove(body, query=None):
    try: return _json(200, _mod("seg").remove(json.loads(body or b"{}").get("id")))
    except (ValueError, AttributeError, OSError) as ex: return _json(400, {"error": str(ex)})


def subject(body, query=None):
    from PIL import Image
    try:
        d = json.loads(body); data = base64.b64decode(d["image"].split(",", 1)[-1])
    except (ValueError, KeyError, TypeError, AttributeError) as ex:
        return _json(400, {"error": f"bad request: {ex}"})
    try:
        sub = _mod("subject")
    except ImportError as ex:   # Vision comes with pyobjc on a Mac; elsewhere the editor finds the subject by itself
        return _json(503, {"error": f"macOS Vision is not available: {ex}", "fallback": "local"})
    t = time.time(); m = sub.mask(data); ms = round((time.time() - t) * 1000)
    return _json(200, {"mask": _png_url(Image.fromarray(m)), "ms": ms})


# Versions of a frame (owner 2026-10-05: Save never overwrites, so undo on the board finds the version it goes back to). Each Save
# writes frame.<n>.json, render.<n>.png and the changed masks and painted layers as <id>.<n>.png; the card points to its version.
# A phase-1 frame (frame.json, render.png) is version 0.
VER = re.compile(r"frame(?:\.([0-9]{1,6}))?\.json")
KEEP = 10


def _frame_dir(d):
    """frames/<stamp> of the library, or ValueError: nothing outside a frame's own folder is listed or removed"""
    if not isinstance(d, str) or not re.fullmatch(r"frames/[A-Za-z0-9_-]+", d): raise ValueError(f"not a frame folder: {d!r}")
    root = os.environ.get("HYIMG_LIBRARY_ROOT") or getattr(sys.modules.get("config"), "W", None)   # the server's library (config.py)
    if not root: raise ValueError("no library root")
    full = os.path.realpath(os.path.join(root, d))
    if not full.startswith(os.path.realpath(os.path.join(root, "frames")) + os.sep): raise ValueError(d)
    return full


def _versions(full):
    out = {}
    for n in os.listdir(full) if os.path.isdir(full) else []:
        m = VER.fullmatch(n)
        if m: out[int(m.group(1) or 0)] = n
    return out


def _refs(doc):
    """the files a frame.json refers to: its render, the painted layers and the masks"""
    out = set()
    if doc.get("render"): out.add(doc["render"])
    def walk(ls):
        for l in ls or []:
            if l.get("file"): out.add(l["file"])
            if isinstance(l.get("mask"), dict) and l["mask"].get("file"): out.add(l["mask"]["file"])
            walk(l.get("children"))
    walk(doc.get("layers"))
    m = doc.get("mask")   # the master mask of the whole frame (owner 2026-10-06): its file and its baked look
    if isinstance(m, dict): out |= {m[k] for k in ("file", "show") if isinstance(m.get(k), str)}
    return out


def versions(body, query=None):
    """POST {"dir": "frames/<stamp>"} -> {"versions": [0, 1, 2], "next": 3}"""
    try: full = _frame_dir(json.loads(body or b"{}").get("dir"))
    except (ValueError, AttributeError) as ex: return _json(400, {"error": str(ex)})
    vs = sorted(_versions(full)); return _json(200, {"versions": vs, "next": (vs[-1] + 1) if vs else 1})


def prune(body, query=None):
    """POST {"dir": "frames/<stamp>", "keep": 10} -> {"kept": [...], "removed": [...]}: the newest versions stay with every file they
    refer to; older frame.<n>.json, their renders and the layer and mask files no kept version needs are removed, only in this folder"""
    try:
        d = json.loads(body or b"{}"); rel = d.get("dir"); full = _frame_dir(rel); keep = max(1, min(50, int(d.get("keep", KEEP))))
    except (ValueError, AttributeError, TypeError) as ex: return _json(400, {"error": str(ex)})
    vs = _versions(full); order = sorted(vs); kept, old = order[-keep:], order[:-keep]
    need = set()
    for n in kept:
        try: need |= _refs(json.load(open(os.path.join(full, vs[n]), encoding="utf-8")))
        except (OSError, ValueError, AttributeError, TypeError): pass   # a json that is not a frame's object: as an unreadable one
        need.add(f"{rel}/render.png" if n == 0 else f"{rel}/render.{n}.png")
    removed = []
    def rm(name):
        p = os.path.join(full, name)
        # the master mask's files stay (owner 2026-10-06): a card's mask, pasted onto pictures, points to them from anywhere on the boards
        if f"{rel}/{name}" in need or not os.path.isfile(p) or name.startswith("masks/main_mask"): return
        os.remove(p); removed.append(name)
    for n in old:
        rm(vs[n]); rm("render.png" if n == 0 else f"render.{n}.png")
    if old:   # the layers and masks only an old version used
        for sub in ("layers", "masks"):
            sd = os.path.join(full, sub)
            for name in os.listdir(sd) if os.path.isdir(sd) else []:
                if re.fullmatch(r"[A-Za-z0-9_-]+(\.[0-9]{1,6})?\.png", name): rm(f"{sub}/{name}")
    return _json(200, {"kept": kept, "removed": removed})


ROUTES = {"inpaint": inpaint, "subject": subject, "versions": versions, "prune": prune, "segment": segment, "matte": matte, "segwarm": segwarm,
          "models": models, "models_remove": models_remove}
