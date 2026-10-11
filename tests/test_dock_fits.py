"""Image Studio in the app's window, the library beside the board (studios audit 2026-10-11). Chromium, dark, Russian.

- S-01 the dock fits: at 1440 × 900 with the library 340 px open, every button of the studio's dock is pressed by the mouse (no library over
  Select) and nothing of the dock lies on the Layers footer; Actions is Hyimg's icon button 36 × 34 with ⌘K in its tooltip, the percent 56 px,
  the switch Board and Image alone, the Hint bar clear of the library. At 1280 × 800 the dock does not fit beside the library: the library folds while the studio is open and
  comes back when it closes
- S-02 the keys typed while the studio is on its way are its own: B pressed right after ↵ opened it gives the brush
- S-20 a board with a frame card loads without an error (grade.js read HY before it was set)

  HYIMG_REPO=<Hyimg checkout> python3 -m pytest tests/test_dock_fits.py
"""
from playwright.sync_api import sync_playwright

from test_imgframe import hy, shot  # noqa: F401  (hy is the fixture)

READY = "() => window.__frames && __frames.ED && __frames.ED.win && __frames.ED.win.__ed && __frames.ED.win.__ed.ready"
# every button of the dock as the person sees it: the element the mouse meets at its middle, through the app's page and the board's frame;
# the dock's box and the Layers footer's, in the window's pixels
SEEN = """() => {
  const f = document.getElementById('cvFrame'), fr = f.getBoundingClientRect(), d = f.contentDocument, ed = d.querySelector('.ifed.on iframe');
  const er = ed.getBoundingClientRect(), foot = ed.contentDocument.getElementById('lbot').getBoundingClientRect();
  const btns = [...d.querySelectorAll('#dock button')].filter(b => b.getClientRects().length && getComputedStyle(b).visibility !== 'hidden'
    && !b.closest('.ifcol'));
  const hit = btns.map(b => { const r = b.getBoundingClientRect(), x = fr.left + r.left + r.width / 2, y = fr.top + r.top + r.height / 2;
    const top = document.elementFromPoint(x, y), inner = top === f ? d.elementFromPoint(x - fr.left, y - fr.top) : null;
    return [b.id || b.dataset.g || b.dataset.a || b.dataset.mode, top === f && !!inner && (inner === b || b.contains(inner))]; });
  const dk = d.getElementById('dock').getBoundingClientRect();
  return { hit, dock: [fr.left + dk.left, fr.top + dk.top, fr.left + dk.right, fr.top + dk.bottom],
    foot: [fr.left + er.left + foot.left, fr.top + er.top + foot.top, fr.left + er.left + foot.right, fr.top + er.top + foot.bottom],
    lib: getComputedStyle(document.querySelector('main')).visibility !== 'hidden' && document.querySelector('main').getBoundingClientRect().width > 0,
    modes: [...d.querySelectorAll('#modes > button')].filter(b => b.getClientRects().length && getComputedStyle(b).display !== 'none').map(b => b.dataset.mode) };
}"""


def app(p, port, w, h):
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": w, "height": h}, color_scheme="dark")
    errors = []; page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"http://127.0.0.1:{port}/?view=panel", wait_until="domcontentloaded")
    page.wait_for_function("() => document.body.classList.contains('cv-on') && !document.body.classList.contains('cv-only')"
                           " && !document.documentElement.classList.contains('lib-wait') && window.hyLibPanel && hyLibPanel.ready", timeout=60000)
    page.evaluate("() => { document.documentElement.style.setProperty('--lw', '340px'); renderFolders(); inset(); }")
    page.wait_for_function("() => Math.round(document.querySelector('main').getBoundingClientRect().width) === 340")
    cv = page.query_selector("#cvFrame").content_frame()
    cv.wait_for_function("() => typeof PLGST !== 'undefined' && PLGST.some(p => p.name === 'frames' && p.ok)", timeout=30000)
    cv.wait_for_selector(".it[data-id=a1]")
    cv.evaluate("() => { cam.x = -800; cam.y = -200; cam.z = 0.5; renderCam(); render(); }"); page.wait_for_timeout(500)   # a1 right of the library
    return browser, page, cv, errors


def enter(page, cv):
    b = cv.locator(".it[data-id=a1]").bounding_box()
    page.mouse.click(b["x"] + 40, b["y"] + 40); page.wait_for_timeout(300); page.keyboard.press("Enter")
    cv.wait_for_function(READY, timeout=30000)
    cv.wait_for_function("() => { const d = document.getElementById('dock'); return d.classList.contains('ifmid') && !d.style.width; }")
    page.wait_for_timeout(1500)   # the library's fold, if any, and its slide


def check(page, where):
    s = page.evaluate(SEEN)
    dead = [k for k, ok in s["hit"] if not ok]
    assert not dead, (where, "dock buttons the mouse does not reach", dead, s)
    d, f = s["dock"], s["foot"]
    assert d[2] <= f[0] or d[0] >= f[2] or d[3] <= f[1] or d[1] >= f[3], (where, "the dock lies on the Layers footer", d, f)
    assert s["modes"] == ["board", "image"], (where, s["modes"])
    return s


def test_the_dock_fits_beside_the_library(hy):
    port, lib, state = hy
    with sync_playwright() as p:
        browser, page, cv, errors = app(p, port, 1440, 900)
        enter(page, cv)
        s = check(page, "1440")
        assert s["lib"], "at 1440 the dock fits beside a 340 px library: it stays open"
        acts = cv.evaluate("() => { const b = document.getElementById('ifActs'), r = b.getBoundingClientRect(); return [Math.round(r.width), Math.round(r.height), b.title, b.textContent.trim()]; }")
        assert acts[:2] == [36, 34] and "⌘K" in acts[2] and acts[3] == "", acts
        assert cv.evaluate("() => Math.round(document.getElementById('ifZoom').getBoundingClientRect().width)") == 56
        # the Hint bar on the studio's page stands clear of the library too (it began under it: «…брать | C Аннотация»)
        bar = page.evaluate("""() => { const ed = document.getElementById('cvFrame').contentDocument.querySelector('.ifed.on iframe'),
          h = ed.contentDocument.querySelector('hy-keyhint[place=top]'); if (!h) return null; const r = h.getBoundingClientRect();
          return [ed.getBoundingClientRect().left + r.left, document.querySelector('main').getBoundingClientRect().right, h.textContent]; }""")
        assert bar and bar[0] >= bar[1], bar
        shot(page, "fits-1440.png")
        cv.evaluate("() => __frames.closeNow()"); cv.wait_for_function("() => !__frames.ED", timeout=20000)
        browser.close()

        browser, page, cv, errors = app(p, port, 1280, 800)
        enter(page, cv)
        s = check(page, "1280")
        assert not s["lib"], "at 1280 the dock does not fit beside the library: it folds while the studio is open"
        shot(page, "fits-1280.png")
        cv.evaluate("() => __frames.closeNow()"); cv.wait_for_function("() => !__frames.ED", timeout=20000)
        page.wait_for_function("() => getComputedStyle(document.querySelector('main')).visibility !== 'hidden' && document.querySelector('main').getBoundingClientRect().width > 300",
                               timeout=5000)   # back as it was
        assert not errors, errors
        browser.close()


def test_keys_typed_while_it_opens_are_the_studios(hy):
    port, lib, state = hy
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 900}, color_scheme="dark")
        errors = []; page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"http://127.0.0.1:{port}/canvas.html")
        page.wait_for_function("() => typeof PLGST !== 'undefined' && PLGST.some(p => p.name === 'frames' && p.ok)", timeout=20000)
        page.wait_for_selector(".it[data-id=a1]")
        page.evaluate("() => { cam.x = -40; cam.y = -40; cam.z = 0.6; renderCam(); render(); }"); page.wait_for_timeout(400)
        b = page.locator(".it[data-id=a1]").bounding_box()
        page.mouse.click(b["x"] + 40, b["y"] + 40); page.wait_for_timeout(300)
        page.keyboard.press("Enter"); page.keyboard.press("KeyB")
        on_its_way = page.evaluate(f"() => MODES.open === 'image' && !({READY})()")
        assert on_its_way, "B came after the studio had opened: the test proves nothing"
        page.wait_for_function(READY, timeout=30000)
        fr = page.query_selector(".ifed.on:not(.away) iframe").content_frame()
        fr.wait_for_function("() => __ed.S.tool === 'brush'", timeout=5000)
        assert page.evaluate("() => document.querySelector('#dock button.ifg.on').dataset.t") == "brush"
        assert not errors, errors
        browser.close()


def test_a_board_with_a_frame_loads_without_errors(hy):
    port, lib, state = hy
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, color_scheme="dark")
        url = f"http://127.0.0.1:{port}/canvas.html"
        page.goto(url)
        page.wait_for_function("() => typeof PLGST !== 'undefined' && PLGST.some(p => p.name === 'frames' && p.ok)", timeout=20000)
        page.wait_for_selector(".it[data-id=a1]")
        page.evaluate("sel = new Set(['b1']); render();"); page.keyboard.press("Alt+Meta+KeyG")
        page.wait_for_function("() => Object.values(board.items).some(it => it.type === 'imgframe')", timeout=20000)
        page.wait_for_function("() => !document.getElementById('save') || !/…|\\.\\.\\./.test(document.getElementById('status')?.textContent || '')")
        page.wait_for_timeout(1500)   # the board is written
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: m.type == "error" and "Failed to load resource" not in m.text and errors.append(m.text))
        page.goto(url)
        page.wait_for_function("() => typeof PLGST !== 'undefined' && PLGST.some(p => p.name === 'frames' && p.ok)", timeout=20000)
        page.wait_for_selector(".plg[data-type=imgframe] img.ifr")
        page.wait_for_timeout(800)
        assert not errors, errors
        browser.close()
