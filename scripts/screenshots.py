"""
Capture README / report screenshots of every page (desktop + mobile, dark + light).

Needs the API (uvicorn backend.app.main:app --port 8000) and the frontend dev
server (npm run dev, port 5173) running.

    python scripts/screenshots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5173"
PAGES = [("landing", "/"), ("compare", "/compare"), ("tamil_nadu", "/tamil-nadu"), ("methodology", "/methodology")]


def settle(page, full: bool = True):
    """Scroll through the page so whileInView reveals fire, then return to top."""
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1500)            # let the route's scroll-to-top settle first
    if full:
        page.mouse.move(700, 450)
        h = page.evaluate("document.documentElement.scrollHeight")
        for _ in range(0, h, 450):
            page.mouse.wheel(0, 450)
            page.wait_for_timeout(160)
        page.wait_for_timeout(900)
        page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(1200)


def set_theme(page, theme: str):
    page.evaluate(f"localStorage.setItem('solarsite-theme', '{theme}')")


def main():
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for theme in ("dark", "light"):
            ctx = browser.new_context(viewport={"width": 1440, "height": 900}, device_scale_factor=1,
                                      color_scheme=theme)
            page = ctx.new_page()
            page.on("console", lambda m: m.type == "error" and errors.append(f"{theme} {page.url}: {m.text}"))
            page.goto(BASE + "/")
            set_theme(page, theme)
            for name, path in PAGES:
                page.goto(BASE + path)
                if name == "compare":
                    page.evaluate("""localStorage.setItem('solarsite-compare', JSON.stringify([
                      {id:'27.5397,71.9157', name:'Bhadla Solar Park, India', lat:27.5397, lon:71.9157},
                      {id:'35.0000,-117.5000', name:'Mojave Desert, USA', lat:35.0, lon:-117.5},
                      {id:'52.3000,13.6000', name:'Brandenburg, Germany', lat:52.3, lon:13.6}]))""")
                    page.reload()
                    page.wait_for_timeout(2500)
                settle(page)
                page.screenshot(path=OUT / f"{name}_{theme}_hero.png")
                page.screenshot(path=OUT / f"{name}_{theme}_full.png", full_page=True)
            # map: overview, then a click-to-analyse on Kamuthi (Tamil Nadu)
            page.goto(BASE + "/map")
            page.wait_for_timeout(3500)
            page.screenshot(path=OUT / f"map_{theme}_overview.png")
            page.goto(BASE + "/map?lat=9.35&lon=78.38")
            page.wait_for_timeout(5000)
            page.screenshot(path=OUT / f"map_{theme}_analysis.png")
            page.locator("aside [class*='overflow-y-auto']").first.evaluate("el => el.scrollTo(0, 900)")
            page.wait_for_timeout(1200)
            page.screenshot(path=OUT / f"map_{theme}_analysis_shap.png")
            ctx.close()

        # mobile (dark)
        ctx = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2, color_scheme="dark",
                                  is_mobile=True, has_touch=True)
        page = ctx.new_page()
        page.on("console", lambda m: m.type == "error" and errors.append(f"mobile {page.url}: {m.text}"))
        page.goto(BASE + "/")
        set_theme(page, "dark")
        page.goto(BASE + "/")
        settle(page)
        page.screenshot(path=OUT / "landing_mobile.png")
        page.goto(BASE + "/map?lat=27.5397&lon=71.9157")
        page.wait_for_timeout(5000)
        page.screenshot(path=OUT / "map_mobile_analysis.png")
        browser.close()

    print(f"saved {len(list(OUT.glob('*.png')))} screenshots to {OUT}")
    print("console errors:" if errors else "no console errors")
    for e in errors[:20]:
        print("  ", e[:300])


if __name__ == "__main__":
    main()
