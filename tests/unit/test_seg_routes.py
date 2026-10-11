"""SAM 3 and BiRefNet behind routes.py (owner 2026-10-10: «давай качай конечно»): segment, matte, segwarm, models, models_remove.
The worker (inpaint/segworker.py) runs as it does for the board, a process of its own, but with HYIMG_SEG_STUB=1 it answers without
models (an ellipse for a word, a disc for a click, the box itself), so these tests take seconds. Missing models answer 503 with the
README's words; Refine Edge without BiRefNet falls back to the guided filter here (edges.py). One test runs the real models and is
skipped where they are not installed."""
import base64
import io
import json
import os
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
from conftest import load  # noqa: E402

REAL = Path(os.path.expanduser("~/Library/Caches/Hyimg/models"))
REAL_PY = Path(os.path.expanduser("~/Library/Caches/Hyimg/venvs/seg/bin/python"))


def png_url(im):
    b = io.BytesIO(); im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def arr(url):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("L"))


def call(routes, name, obj):
    code, ctype, data = routes.ROUTES[name](json.dumps(obj).encode())
    assert ctype == "application/json"
    return code, json.loads(data)


def scene(w=320, h=240):
    """a light disc on a dark ground"""
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.where((xx - 160) ** 2 + (yy - 120) ** 2 <= 70 ** 2, 230, 30).astype(np.uint8)
    return Image.fromarray(np.dstack([a, a, a]))


@pytest.fixture
def stubbed(routes, tmp_path, monkeypatch):
    monkeypatch.setenv("HYIMG_SEG_STUB", "1")
    monkeypatch.setenv("HYIMG_SEG_PYTHON", sys.executable)
    monkeypatch.setenv("HYIMG_MODELS", str(tmp_path / "models"))
    monkeypatch.setenv("HYIMG_SEG_IDLE", "30")
    yield routes
    routes._mod("seg").WORKER.stop()


@pytest.fixture
def bare(routes, tmp_path, monkeypatch):
    monkeypatch.setenv("HYIMG_MODELS", str(tmp_path / "models"))
    monkeypatch.setenv("HYIMG_SEG_PYTHON", str(tmp_path / "nowhere/python"))
    return routes


def test_segment_by_word_click_and_box_through_the_worker(stubbed):
    url = png_url(scene())
    code, r = call(stubbed, "segment", {"image": url, "text": "disc"})
    assert code == 200 and r["how"] == "text" and len(r["masks"]) == len(r["boxes"]) == len(r["scores"]) == 1
    m = arr(r["masks"][0]); assert m.shape == (240, 320) and m[120, 160] == 255 and m[5, 5] == 0
    code, r = call(stubbed, "segment", {"image": url, "points": [{"x": 60, "y": 50, "label": 1}]})
    m = arr(r["masks"][0]); assert code == 200 and r["how"] == "points" and m[50, 60] == 255 and m[200, 300] == 0
    code, r = call(stubbed, "segment", {"image": url, "box": {"x": 10, "y": 20, "w": 50, "h": 40}})
    assert code == 200 and r["how"] == "box" and r["boxes"][0] == [10, 20, 50, 40]
    # the same worker answers them all: one process, kept warm
    assert stubbed._mod("seg").WORKER.alive()


def test_bad_requests_say_why(stubbed):
    assert call(stubbed, "segment", {"image": png_url(scene())})[0] == 400                      # no word, click or box
    assert call(stubbed, "segment", {"text": "x"})[0] == 400                                     # no picture
    assert call(stubbed, "matte", {"image": png_url(scene()), "method": "magic"})[0] == 400


def test_matte_with_a_mask_and_alone(stubbed):
    url, mask = png_url(scene()), Image.new("L", (320, 240), 0)
    mask.paste(255, (100, 60, 220, 180))
    code, r = call(stubbed, "matte", {"image": url, "mask": png_url(mask)})
    a = arr(r["alpha"]); assert code == 200 and r["how"] == "birefnet" and a[120, 160] > 200 and a[5, 5] < 10
    code, r = call(stubbed, "matte", {"image": url})   # Remove Background: the whole picture
    assert code == 200 and arr(r["alpha"]).shape == (240, 320)


def test_missing_models_answer_503_with_the_readme(bare):
    code, r = call(bare, "segment", {"image": png_url(scene()), "text": "disc"})
    assert code == 503 and r["missing"] == "sam3-4bit" and "README" in r["error"]
    code, r = call(bare, "matte", {"image": png_url(scene())})
    assert code == 503 and r["fallback"] == "vision"   # Remove Background goes back to macOS Vision
    code, r = call(bare, "segwarm", {"models": ["sam", "bir"]})
    assert code == 503
    st = call(bare, "models", {})[1]
    assert [m["id"] for m in st["models"]] == ["lama", "sam3-4bit", "birefnet-lite"] and not any(m["installed"] for m in st["models"])
    assert st["runtime"]["installed"] is False


def test_refine_without_birefnet_is_the_guided_filter(bare):
    """a coarse mask, 3 px inside the disc's edge (the guided filter's reach here is 4 px), comes out on the edge"""
    im, mask = scene(), Image.new("L", (320, 240), 0)
    from PIL import ImageDraw
    ImageDraw.Draw(mask).ellipse((93, 53, 227, 187), fill=255)   # r = 67, the disc's is 70
    code, r = call(bare, "matte", {"image": png_url(im), "mask": png_url(mask)})
    a = arr(r["alpha"])
    assert code == 200 and r["how"] == "guided" and "BiRefNet" in r["note"]
    assert a[120, 160 + 69] > 128 > np.asarray(mask)[120, 160 + 69]   # inside the disc, outside the coarse mask: taken
    assert a[120, 160 + 76] < 60 and a[120, 160] > 240


def test_settings_lists_sizes_and_remove_takes_only_a_models_folder(bare, tmp_path):
    d = tmp_path / "models" / "birefnet-lite"; d.mkdir(parents=True); (d / "model.safetensors").write_bytes(b"x" * 1000)
    st = call(bare, "models", {})[1]
    b = next(m for m in st["models"] if m["id"] == "birefnet-lite")
    assert b["installed"] and b["bytes"] == 1000 and b["licence"] == "MIT"
    for bad in ("../x", "venvs", None):
        assert call(bare, "models_remove", {"id": bad})[0] == 400
    (tmp_path / "models" / "sam3-4bit").symlink_to(tmp_path)   # a link out of the folder is not followed
    assert call(bare, "models_remove", {"id": "sam3-4bit"})[0] == 400 and tmp_path.exists()
    code, r = call(bare, "models_remove", {"id": "birefnet-lite"})
    assert code == 200 and r["freed"] == 1000 and not d.exists()


def test_edges_guided_keeps_a_mask_with_no_edge_alone():
    edges = load("edges", "hyimg_frames_edges_unit")
    rgb = np.zeros((40, 40, 3), np.uint8)
    assert (edges.guided(rgb, np.zeros((40, 40), np.uint8)) == 0).all()
    assert (edges.guided(rgb, np.full((40, 40), 255, np.uint8)) == 255).all()
    assert edges.radius(1856, 2304) == 12


@pytest.mark.skipif(not ((REAL / "sam3-4bit/model.safetensors").is_file() and (REAL / "birefnet-lite/model.safetensors").is_file()
                         and REAL_PY.is_file()), reason="SAM 3, BiRefNet lite or their runtime not installed (README «Models»)")
def test_real_models_find_the_disc_and_refine_it(routes, monkeypatch):
    """the real SAM 3 and BiRefNet on this Mac: slow on the first call (the models load), read-only on the models' folder"""
    monkeypatch.setenv("HYIMG_SEG_IDLE", "60")
    try:
        url = png_url(scene(640, 480).resize((640, 480)))
        code, r = call(routes, "segment", {"image": url, "points": [{"x": 320, "y": 240, "label": 1}]})
        assert code == 200, r
        m = arr(r["masks"][0]); assert m[240, 320] > 127 and m[10, 10] < 128
        code, a = call(routes, "matte", {"image": url, "mask": r["masks"][0]})
        assert code == 200 and a["how"] == "birefnet", a
        assert arr(a["alpha"])[240, 320] > 200
    finally:
        routes._mod("seg").WORKER.stop()
