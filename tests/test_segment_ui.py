"""SAM 3 and BiRefNet in Image Studio (owner 2026-10-10: «давай качай конечно»): the Smart Select tool beside the wand (⇧W, the rail's
and the dock's selection group), its options bar («Select by text…», the LED, Refine Edge), its click, the Select menu's new items, the
agent's commands (select.prompt, select.click, mask.refine, matte.export), Remove Background through BiRefNet, Settings › Image Studio,
and the words when a model is missing. The plugin's worker answers without models here (HYIMG_SEG_STUB=1, inpaint/segworker.py: a word
selects a centred ellipse, a click a disc), so this is the interface's path, not the models' quality (tests/unit/test_seg_routes.py has
a real-model test). Chromium, dark, its own server on a free port and a temporary library."""
import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid

import pytest

from editor_kit import at, focus, settle
from test_imgframe import HERE, ITEMS, REPO, editor, free_port, open_board, picture
from test_select_tool import open_editor


def serve(tmp_path, stub):
    lib, state, plugins = tmp_path / "lib", tmp_path / "state", tmp_path / "plugins"
    (lib / "pics").mkdir(parents=True); (state / "boards").mkdir(parents=True); plugins.mkdir(); (plugins / "frames").symlink_to(HERE)
    picture(600, 400, (200, 40, 40), (240, 180, 60)).save(lib / "pics/a.png")
    picture(300, 450, (30, 60, 200), (60, 200, 220)).save(lib / "pics/b.jpg", quality=92)
    picture(2400, 1600, (40, 160, 90), (200, 220, 90)).save(lib / "pics/big.jpg", quality=90)
    (state / "boards/main.json").write_text(json.dumps({"items": ITEMS, "groups": {}, "removed": {}, "revision": 1}))
    (tmp_path / "settings.json").write_text("{}")
    port = free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HYIMG_", "REVIEW_"))}
    env.update(HYIMG_LIBRARY_ROOT=str(lib), HYIMG_STATE_ROOT=str(state), HYIMG_PROJECT_ID=str(uuid.uuid4()), HYIMG_PLUGINS=str(plugins),
               HYIMG_SETTINGS=str(tmp_path / "settings.json"), PYTHONDONTWRITEBYTECODE="1", HYIMG_MODELS=str(tmp_path / "models"),
               HYIMG_SEG_IDLE="60", HYIMG_SEG_PYTHON=sys.executable if stub else str(tmp_path / "no-runtime/python"))
    if stub: env["HYIMG_SEG_STUB"] = "1"
    log = open(tmp_path / "server.log", "w+")
    proc = subprocess.Popen([sys.executable, str(REPO / "review/server.py"), str(port)], env=env, stdout=log, stderr=log)
    for _ in range(100):
        try: urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1); break
        except OSError: time.sleep(0.1)
    return proc, log, port


@pytest.fixture
def seg_hy(tmp_path, request):
    proc, log, port = serve(tmp_path, request.param == "stub")
    try: yield port
    finally: proc.terminate(); proc.wait(5); log.close()


SEL = "() => hyImage.info().selection"


@pytest.mark.parametrize("seg_hy", ["stub"], indirect=True)
def test_smart_select_tool_bar_click_menu_and_ops(seg_hy):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser, page, fr, errors = open_editor(p, seg_hy, "chromium")
        page.emulate_media(color_scheme="dark"); settle(fr, 600)
        # the tool: beside the wand in the rail's group and in the board dock's, ⇧W from the wand, its own icon
        assert fr.evaluate("() => __ed.GROUPS.find(g => g.tools.includes('wand')).tools.includes('smart')")
        assert "smart" in [t["id"] for g in fr.evaluate("() => hyEdK.dock.groups()") for t in g["tools"]]
        assert fr.evaluate("() => hyEdK.svg('smart').includes('<svg')")
        focus(page, fr); page.keyboard.press("KeyW"); settle(fr); page.keyboard.press("Shift+KeyW"); settle(fr)
        assert fr.evaluate("() => __ed.S.tool") == "smart"
        # its options bar: the mode, «Select by text…», the LED, Refine Edge
        assert fr.locator("#obar .segq").count() == 1 and fr.locator("#obar hy-led.segled").count() == 1
        assert fr.locator("#obar .segq").get_attribute("placeholder") == "Select by text…"
        # a word and ↵: the stub's ellipse in the middle of the 1000 × 450 document
        fr.locator("#obar .segq").fill("disc"); fr.locator("#obar .segq").press("Enter")
        fr.wait_for_function("() => __ed.S.hasSel && !hyEdK.seg.busy", timeout=30000); settle(fr)
        s = fr.evaluate(SEL); assert abs(s["x"] - 250) < 12 and abs(s["w"] - 500) < 20 and abs(s["y"] - 112) < 12, s
        assert fr.evaluate("() => document.querySelector('#obar hy-led.segled').getAttribute('state')") == "off"
        # a click on the picture: the object under it (the stub's disc around the point), the selection's mode new
        page.mouse.click(*at(page, fr, 150, 120))
        fr.wait_for_function("s => { const b = hyImage.info().selection; return b && b.w < 200; }", arg=None, timeout=30000)
        s = fr.evaluate(SEL); assert abs(s["x"] + s["w"] / 2 - 150) < 8 and abs(s["y"] + s["h"] / 2 - 120) < 8, s
        # the Select menu: Select by Text… and Smart Select after Subject, Refine Edge
        menu = fr.evaluate("() => hyEdK.MB.find(m => m[0] === 'select')[2]().filter(Boolean).map(i => i.label)")
        assert menu.index("Select by Text…") == menu.index("Subject") + 1 and "Smart Select Tool" in menu and menu[-1] == "Refine Edge"
        # the agent's commands, the same functions
        r = fr.evaluate("() => hyImage.run([{op: 'select.prompt', text: 'disc', pick: 'all'}, {op: 'mask.refine'}, {op: 'layer.select', layer: 'top'},"
                        " {op: 'matte.export', as: 'layer'}, {op: 'select.click', points: [{x: 700, y: 300, label: 1}], mode: 'add'}])")
        assert r["ok"], r
        assert r["results"][0]["count"] == 1 and r["results"][1]["how"] == "birefnet" and r["results"][3]["step"]
        assert fr.evaluate("() => hyEdK.one().name") == "Cut-out"
        assert {"select.prompt", "select.click", "mask.refine", "matte.export"} <= set(fr.evaluate("() => Object.keys(hyImage.OPS)"))
        png = fr.evaluate("() => hyImage.run([{op: 'matte.export', as: 'png'}])")
        assert png["ok"] and png["results"][0]["alpha"].startswith("data:image/png") and png["results"][0]["size"] == [1000, 450]
        # Remove Background: BiRefNet (the stub's) instead of Vision
        r = fr.evaluate("() => hyImage.run([{op: 'select.none'}, {op: 'removeBackground', layer: 'base'}])")
        assert r["ok"], r
        assert fr.evaluate("() => window.__lastRemoveBg.how") == "birefnet"
        assert not errors, errors
        browser.close()


@pytest.mark.parametrize("seg_hy", ["missing"], indirect=True)
def test_missing_models_say_so_and_settings_lists_them(seg_hy):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser, page, errors = open_board(p, seg_hy)
        page.emulate_media(color_scheme="dark")
        # Settings › Image Studio: the three models, none installed, the runtime, the README's words; its rail entry on the board
        page.click("#bset"); page.wait_for_selector("#sets.open")
        page.wait_for_function("() => document.querySelectorAll('#hyImgModels .hs-row').length === 8", timeout=10000)
        txt = page.inner_text("#hyImgModels")
        assert "SAM 3 (MLX, 4-bit)" in txt and "BiRefNet lite" in txt and "LaMa" in txt and txt.count("Not installed") == 4 and "README" in txt
        assert page.evaluate("() => { const b = document.querySelector('.sw-rail [data-go=imagestudio]'); return !!b && !b.hidden; }")
        page.keyboard.press("Escape"); page.wait_for_timeout(300)
        page.evaluate("sel = new Set(['a1', 'b1']); render(); fit()"); time.sleep(0.6)
        page.keyboard.press("Alt+Meta+KeyG")
        page.wait_for_function("() => Object.values(board.items).some(it => it.type === 'imgframe')", timeout=20000)
        fid = page.evaluate("() => Object.entries(board.items).find(([k, it]) => it.type === 'imgframe')[0]")
        page.wait_for_timeout(1200); page.dblclick(f".plg[data-id='{fid}']"); fr = editor(page)
        # a word with no SAM 3: the command fails with where to get it; Remove Background still works (Vision or the studio's own)
        r = fr.evaluate("() => hyImage.run([{op: 'select.prompt', text: 'disc'}])")
        assert not r["ok"] and "SAM 3 (MLX, 4-bit) is not installed" in r["error"] and "README" in r["error"]
        r = fr.evaluate("() => hyImage.run([{op: 'mask.refine'}])"); assert not r["ok"] and "selection" in r["error"].lower()
        r = fr.evaluate("() => hyImage.run([{op: 'removeBackground', layer: 'base'}])")
        assert r["ok"], r
        assert fr.evaluate("() => window.__lastRemoveBg.how") in ("vision", "local")
        # Refine Edge without BiRefNet: the guided filter on the server
        r = fr.evaluate("() => hyImage.run([{op: 'select.rect', x: 100, y: 100, w: 300, h: 200}, {op: 'mask.refine'}])")
        assert r["ok"] and r["results"][1]["how"] == "guided", r
        assert not errors, errors
        browser.close()
