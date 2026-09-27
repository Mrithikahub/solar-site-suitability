"""
End-to-end check of: curated solar parks in search, satellite auto-switch
(and restore on area searches), famous-park chips, and the exclusion card.

    python scripts/test_search_extras.py
"""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://localhost:5173"
OUT = Path(__file__).resolve().parent.parent / "report" / "screenshots"


def settle(page):
    prev = None
    for _ in range(40):
        page.wait_for_timeout(600)
        cur = page.evaluate("(() => { const m = window.__solarMap; const c = m.getCenter(); return [m.getZoom(), +c.lat.toFixed(5), +c.lng.toFixed(5)] })()")
        if cur == prev and float(cur[0]).is_integer():
            return cur
        prev = cur
    return prev


def basemap(page):
    return page.evaluate("[...document.querySelectorAll('.leaflet-tile-pane img')].some(i => i.src.includes('World_Imagery')) ? 'satellite' : 'map'")


def search(page, q):
    box = page.get_by_role("combobox")
    box.click()
    box.fill(q)
    box.press("Enter")


def wait_analysis(page):
    page.wait_for_selector("aside[aria-label='Site analysis']", timeout=20000)
    page.wait_for_function("!document.querySelector('[aria-busy]')", timeout=30000)


def main():
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_context(viewport={"width": 1440, "height": 900}, color_scheme="dark").new_page()
        page.goto(BASE + "/map")
        page.wait_for_timeout(2500)
        print("start basemap:", basemap(page))
        page.screenshot(path=OUT / "extras_chips.png")

        # 1) "Kamuthi" -> curated solar park, satellite
        search(page, "Kamuthi")
        wait_analysis(page)
        v = settle(page)
        page.wait_for_timeout(1500)
        name = page.get_by_role("combobox").input_value()
        print(f"Kamuthi        -> box='{name}' view={v} basemap={basemap(page)} url={page.url.split('?')[1]}")
        page.screenshot(path=OUT / "extras_kamuthi_satellite.png")

        # 2) area search restores the previous basemap
        page.keyboard.press("Escape")
        before = page.evaluate("window.__solarMap.getZoom()")
        search(page, "Nevada")
        page.wait_for_function(f"window.__solarMap.getZoom() !== {before}", timeout=20000)
        v = settle(page)
        page.wait_for_timeout(1200)
        print(f"Nevada         -> view={v} basemap={basemap(page)}")

        # 3) chip click
        page.get_by_role("button", name="Benban").click()
        wait_analysis(page)
        v = settle(page)
        page.wait_for_timeout(1500)
        print(f"Chip Benban    -> view={v} basemap={basemap(page)} url={page.url.split('?')[1]}")
        page.screenshot(path=OUT / "extras_benban_chip.png")

        # 4) excluded site message
        page.keyboard.press("Escape")
        search(page, "Lords stadium London")
        wait_analysis(page)
        settle(page)
        page.wait_for_timeout(1200)
        msg = page.locator("aside [role=status]").first.inner_text()
        print("Lords excluded ->", msg.replace("\n", " | "))
        page.screenshot(path=OUT / "extras_excluded_urban.png")
        b.close()


if __name__ == "__main__":
    main()
