"""Image Studio has no «Shortcuts in tooltips» switch of its own: a tooltip always names its key, as Mac menus and Figma do (owner
2026-10-10, round 18 question 19, r18-small.html, version B: «Да, все окей, давай так»). The board has its own «Key hints» setting, one
setting was enough. The settings popover (the gear of the editor alone) keeps the paper and its note and says nothing of shortcuts;
a tooltip of a control with a key shows the key, also on a page that still wears the old switch's class (body.nokeys). Chromium,
dark, the mouse.

  HYIMG_REPO=<Hyimg checkout> python3 -m pytest tests/test_key_tips.py
"""
from test_imgframe import REPO, hy  # noqa: F401  (hy is the fixture)
from test_studio_dock import open_studio


def test_tooltips_always_name_their_key(hy):
    from playwright.sync_api import sync_playwright
    port, lib, state = hy
    with sync_playwright() as p:
        browser, page, fr, errors = open_studio(p, port)
        # the gear's popover (shown when the editor runs alone, a Studio hides it and the board's Settings speak)
        words = fr.evaluate("() => document.querySelector('#sets').textContent")
        assert "Бумага" in words or "Paper" in words, words
        for gone in ("Shortcuts in tooltips", "Клавиши в подсказках", "Interface", "Интерфейс"):
            assert gone not in words, words
        assert fr.locator("#sets .tg, #setTips").count() == 0
        # a control with a key: its tooltip shows the key, the old switch's class changes nothing
        fr.evaluate("() => document.body.classList.add('nokeys')")
        el = fr.locator("[data-tip][data-key]:visible").first
        key = el.get_attribute("data-key")
        el.hover(); fr.wait_for_selector("#hytip[shown] hy-kbd", timeout=3000)   # Hyimg's one tooltip (ui/hy/tooltip.js, round 20): a cap a key
        got = fr.evaluate("() => [...document.querySelectorAll('#hytip hy-kbd')].map(k => [k.textContent, getComputedStyle(k).display !== 'none' && k.offsetWidth > 0])")
        assert "".join(k for k, _ in got) == key.replace(" ", "") and all(v for _, v in got), (key, got)
        assert not errors, errors
        browser.close()
