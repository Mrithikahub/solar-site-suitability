"""
End-to-end check of map search: type a query, press Enter (no selection), and
verify where the map lands, whether a marker + analysis opened, and the score.

    python scripts/test_search.py
"""
from __future__ import annotations

import math
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://localhost:5173"
OUT = Path(__file__).resolve().parent.parent / "report" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

CASES = [
    # query, expected kind, expected (lat, lon), tolerance km
    ("Lords stadium London", "point", (51.5297, -0.1722), 0.3),
    ("Eiffel Tower", "point", (48.8584, 2.2945), 0.3),
    ("Bhadla Solar Park", "point", (27.538, 71.917), 3.0),
    ("Kamuthi", "area", (9.409, 78.368), 5.0),
    ("Taj Mahal", "point", (27.1751, 78.0421), 0.3),
    ("Nevada", "area", (39.5, -116.9), 250.0),
    ("51.5294, -0.1727", "point", (51.5294, -0.1727), 0.05),
]


def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def main():
    rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, color_scheme="dark")
        page = ctx.new_page()

        # 1) autocomplete dropdown screenshot
        page.goto(BASE + "/map")
        page.wait_for_timeout(2500)
        box = page.get_by_role("combobox")
        box.click()
        box.type("Lords stadium London", delay=35)
        page.wait_for_selector("[role=listbox] [role=option]", timeout=25000)
        page.keyboard.press("ArrowDown")
        page.wait_for_timeout(600)
        page.screenshot(path=OUT / "search_autocomplete.png")

        # 2) each query: type + Enter immediately (no selection)
        for q, kind, exp, tol in CASES:
            page.goto(BASE + "/map")
            page.wait_for_timeout(2000)
            box = page.get_by_role("combobox")
            box.click()
            box.fill(q)
            box.press("Enter")
            if kind == "point":
                page.wait_for_function("new URLSearchParams(location.search).has('lat')", timeout=30000)
                page.wait_for_selector("aside[aria-label='Site analysis']", timeout=15000)
                page.wait_for_function("!document.querySelector('[aria-busy]')", timeout=30000)
            # wait until the fly animation has finished (view stable for 0.6 s)
            prev = None
            for _ in range(40):
                page.wait_for_timeout(600)
                cur = page.evaluate("(() => { const m = window.__solarMap; const c = m.getCenter(); return [m.getZoom(), +c.lat.toFixed(5), +c.lng.toFixed(5)] })()")
                if cur == prev and float(cur[0]).is_integer():
                    break
                prev = cur
            page.wait_for_timeout(800)
            st = page.evaluate("""() => {
                const m = window.__solarMap; const c = m.getCenter(); const u = new URLSearchParams(location.search)
                const panel = document.querySelector("aside[aria-label='Site analysis']")
                const score = panel ? (panel.querySelector('[role=img]')?.getAttribute('aria-label') || '') : ''
                const src = panel ? (panel.innerText.includes('Live Earth Engine') ? 'live' : panel.innerText.includes('Grid estimate') ? 'grid' : '') : ''
                return {zoom: m.getZoom(), center: [c.lat, c.lng], marker: u.has('lat') ? [+u.get('lat'), +u.get('lon')] : null,
                        panel: !!panel, score, src, bounds: m.getBounds().toBBoxString()}
            }""")
            where = st["marker"] if kind == "point" else st["center"]
            d = km(where, exp) if where else float("nan")
            ok = (d <= tol) and (st["panel"] == (kind == "point"))
            rows.append((q, kind, st["zoom"], where, round(d, 3), st["panel"], st["score"], st["src"], ok))
            slug = q.lower().replace(" ", "_").replace(",", "").replace(".", "")[:24]
            page.screenshot(path=OUT / f"search_{slug}.png")
        browser.close()

    print(f"{'query':24s} {'kind':6s} {'zoom':>4s}  {'landed at':22s} {'err km':>7s}  panel  score / source")
    for q, kind, z, where, d, panel, score, src, ok in rows:
        w = f"{where[0]:.4f}, {where[1]:.4f}" if where else "-"
        print(f"{q:24s} {kind:6s} {z:>4.1f}  {w:22s} {d:7.3f}  {str(panel):5s}  {score} {src}  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
