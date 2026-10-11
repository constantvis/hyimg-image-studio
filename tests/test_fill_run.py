"""A fill's layer keeps how long it took and where it ran (owner 2026-10-11: the time of a generation or a render is required, «и compute
power какой-то условный»): Content-Aware Fill run by an agent (hy.py image run … inpaint) saves the frame, and its new layer in frame.N.json
has run {seconds, compute}; a reopened frame keeps it. Chromium, the plugin's own server fixture (test_imgframe.hy)."""
import json
import subprocess
import sys

import pytest

from test_imgframe import REPO, board, hy  # noqa: F401  (the fixture)


@pytest.mark.parametrize("hy", ["en"], indirect=True)
def test_a_fill_layer_keeps_its_seconds_and_compute(hy):
    port, lib, state = hy
    hyp = lambda *a: subprocess.run([sys.executable, str(REPO / "review/hy.py"), *a, "--port", str(port), "--page", "main"],
                                    capture_output=True, text=True, timeout=240)
    out = hyp("do", 'frame a1 name="Pic"', "--quiet"); assert out.returncode == 0, out.stdout + out.stderr
    ops = [{"op": "select.rect", "x": 200, "y": 100, "w": 120, "h": 90}, {"op": "inpaint"}, {"op": "select.none"}]
    out = hyp("image", "run", "Pic", json.dumps(ops)); assert out.returncode == 0, out.stdout + out.stderr
    card = next(it for it in board(state)["items"].values() if it.get("type") == "imgframe")
    doc = json.loads((lib / card["doc"]).read_text())
    fill = [l for l in doc["layers"] if l.get("kind") == "paint" and l.get("run")]
    assert len(fill) == 1, doc["layers"]
    run = fill[0]["run"]
    assert isinstance(run["seconds"], (int, float)) and 0 < run["seconds"] < 240 and run["compute"].startswith("local · "), run
    # the next run opens that version and saves again: the layer still says how it was made
    out = hyp("image", "run", "Pic", '[{"op": "layer.new", "name": "Empty"}]'); assert out.returncode == 0, out.stdout + out.stderr
    card = next(it for it in board(state)["items"].values() if it.get("type") == "imgframe")
    doc = json.loads((lib / card["doc"]).read_text())
    assert [l["run"] for l in doc["layers"] if l.get("run")] == [run], doc["layers"]
