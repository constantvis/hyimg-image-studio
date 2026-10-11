"""A mask's edge put onto the picture's own edges, numpy only (the board's server and the segmentation worker both use it).

  guided(rgb, mask, r)            the guided filter (He et al. 2010) of a mask with the picture as the guide: a coarse mask (SAM 3's
                                  come at 288 px) snaps to the edges in the picture, fingers, hair, the chrome of a frame; a soft edge
                                  stays where the picture is soft; a smoothstep from 0.2 to 0.6 firms the ramp the filter leaves.
                                  Works on the box around the mask's edge only
  refine(rgb, mask, r, matte)     the edge Image Studio keeps: guided; with a BiRefNet alpha (matte) also cut where BiRefNet sees
                                  background, so an edge on the sky or a wall comes out crisp, while the edge between the object and
                                  what covers it (fingers over a phone: both are foreground to BiRefNet) stays the guided one
  radius(w, h)                    the band SAM's resolution leaves: about 1.5 of its pixels at this picture's size
"""
import numpy as np


def radius(w, h):
    return int(max(4, round(max(w, h) / 288 * 1.5)))


def _box(a, r):
    """the mean over a (2r+1)² square, edges repeated (summed areas)"""
    p = np.pad(a, ((r + 1, r), (r + 1, r)), mode="edge")
    c = p.cumsum(0).cumsum(1)
    n = 2 * r + 1
    return (c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]) / (n * n)


def guided(rgb, mask, r=None, eps=1e-3):
    """mask: H × W uint8 (255 = in); returns H × W uint8, only the box around the mask's edge recomputed"""
    m = np.asarray(mask)
    h, w = m.shape
    r = r or radius(w, h)
    on = m > 127
    if not on.any() or on.all():
        return m.copy()
    ys, xs = np.nonzero(on)
    pad = 3 * r
    y0, y1, x0, x1 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad), max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
    I = np.asarray(rgb, np.float32)[y0:y1, x0:x1, :3] @ np.array([0.299, 0.587, 0.114], np.float32) / 255
    p = m[y0:y1, x0:x1].astype(np.float32) / 255
    mI, mp = _box(I, r), _box(p, r)
    a = (_box(I * p, r) - mI * mp) / (_box(I * I, r) - mI * mI + eps)
    b = mp - a * mI
    q = _box(a, r) * I + _box(b, r)
    t = np.clip((q - 0.2) / 0.4, 0, 1)   # the filter leaves a ramp along the edge: its upper part is the object, a smoothstep firms it
    out = m.copy()
    out[y0:y1, x0:x1] = np.rint(t * t * (3 - 2 * t) * 255).astype(np.uint8)
    return out


def refine(rgb, mask, r=None, matte=None):
    """guided, and with BiRefNet's alpha (H × W uint8) the background it sees taken away near the edge"""
    m = np.asarray(mask)
    r = r or radius(m.shape[1], m.shape[0])
    q = guided(rgb, m, r)
    if matte is None:
        return q
    core = _box((m > 127).astype(np.float32), r) > 0.999   # deep inside: the object, whatever BiRefNet says
    a = np.asarray(matte)
    return np.where(core, q, np.minimum(q, a)).astype(np.uint8)
