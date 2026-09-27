"""
End-to-end smoke test of the deployed site (Vercel frontend + Render API).

    python scripts/test_live.py [site_url]
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

SITE = (sys.argv[1] if len(sys.argv) > 1 else "https://solar-site-suitability.vercel.app").rstrip("/")
OUT = Path(__file__).resolve().parent.parent / "report" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    errors, failed = [], []
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": 1440, "height": 900}, color_scheme="dark", accept_downloads=True)
        page = ctx.new_page()
        page.on("console", lambda m: m.type == "error" and errors.append(m.text[:200]))
        page.on("requestfailed", lambda r: failed.append(f"{r.url[:90]} {r.failure}"))
        page.on("response", lambda r: r.status >= 400 and "onrender" in r.url and failed.append(f"{r.status} {r.url[:90]}"))

        # landing
        page.goto(SITE + "/", wait_until="networkidle")
        print("landing:", page.title())

        # map: offline banner must be gone, layers come from the API
        page.goto(SITE + "/map", wait_until="networkidle")
        page.wait_for_timeout(3000)
        offline = page.locator("text=Analysis server offline").count()
        overlays = page.evaluate("[...document.querySelectorAll('.leaflet-image-layer')].map(i => i.complete && i.naturalWidth > 0)")
        print(f"map: offline banner={bool(offline)}  score overlay loaded={overlays}")

        # famous-park chip -> analysis
        page.get_by_role("button", name="Bhadla").click()
        page.wait_for_selector("aside[aria-label='Site analysis']", timeout=60000)
        page.wait_for_function("!document.querySelector('[aria-busy]')", timeout=90000)
        page.wait_for_timeout(2500)
        panel = page.locator("aside[aria-label='Site analysis']").inner_text()
        gauge = page.locator("aside [role=img]").first.get_attribute("aria-label")
        src = "live" if "Live Earth Engine" in panel else "grid" if "Grid estimate" in panel else "?"
        print(f"chip Bhadla: {gauge}  source={src}")
        page.screenshot(path=OUT / "live_map_bhadla.png")

        # PDF report download
        with page.expect_download(timeout=90000) as dl:
            page.get_by_role("link", name="Download report").click()
        d = dl.value
        path = OUT / "live_report.pdf"
        d.save_as(path)
        print(f"report: {d.suggested_filename} {path.stat().st_size // 1024} KB")

        # search: POI -> analysis
        page.keyboard.press("Escape")
        box = page.get_by_role("combobox")
        box.click()
        box.fill("Eiffel Tower")
        box.press("Enter")
        page.wait_for_function("new URLSearchParams(location.search).get('lat')?.startsWith('48.85')", timeout=60000)
        page.wait_for_function("!document.querySelector('[aria-busy]')", timeout=90000)
        page.wait_for_timeout(2000)
        panel = page.locator("aside[aria-label='Site analysis']").inner_text()
        if page.locator("aside [role=status]").count():
            result = page.locator("aside [role=status]").first.inner_text().splitlines()[0]
        else:
            result = page.locator("aside [role=img]").first.get_attribute("aria-label")
        src = "live" if "Live Earth Engine" in panel else "grid" if "Grid estimate" in panel else "?"
        print(f"search Eiffel Tower: {page.url.split('?')[1]}  -> {result}  source={src}")

        # compare + methodology
        page.goto(SITE + "/compare", wait_until="networkidle")
        page.get_by_role("button", name="Load an example").click()
        page.wait_for_selector("text=Scores side by side", timeout=90000)
        print("compare: example loaded, charts rendered")
        page.goto(SITE + "/methodology#case-study", wait_until="networkidle")
        page.wait_for_timeout(3000)
        print("methodology: case-study top =", round(page.evaluate("document.getElementById('case-study').getBoundingClientRect().top")))
        page.goto(SITE + "/map", wait_until="networkidle")
        page.wait_for_timeout(2500)
        page.screenshot(path=OUT / "live_map.png")
        b.close()

    print("console errors:", errors[:5] if errors else "none")
    print("failed API requests:", failed[:5] if failed else "none")


if __name__ == "__main__":
    main()
