"""
Copy static assets the frontend ships with (so landing / methodology / case-study
pages render without the API):

  report/figures/*.png            -> frontend/public/figures/*.webp
  data/processed/layers/*.webp    -> frontend/public/img/layers/ (landing + case study)
  metrics / AHP / physical JSON   -> frontend/public/data/

    python scripts/sync_frontend_assets.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "frontend" / "public"
FIG_OUT = PUBLIC / "figures"
IMG_OUT = PUBLIC / "img" / "layers"
DATA_OUT = PUBLIC / "data"
for d in (FIG_OUT, IMG_OUT, DATA_OUT):
    d.mkdir(parents=True, exist_ok=True)

# report figures -> webp (white-ish figure surface kept, no alpha needed)
for png in sorted((ROOT / "report" / "figures").glob("*.png")):
    im = Image.open(png).convert("RGB")
    if im.width > 2400:
        im = im.resize((2400, round(im.height * 2400 / im.width)), Image.LANCZOS)
    im.save(FIG_OUT / f"{png.stem}.webp", "WEBP", quality=86, method=6)

# layer imagery used on static pages
LAYERS = ["truecolor", "ndvi", "ndbi", "lst", "slope", "elevation", "landcover", "tn_ghi", "ml_tn", "ahp_tn",
          "physical_global", "ml_global", "ahp_global", "global_ghi", "global_ndvi", "global_landcover",
          "global_slope", "global_lst", "global_nightlights", "global_population"]
for name in LAYERS:
    src = ROOT / "data" / "processed" / "layers" / f"{name}.webp"
    shutil.copy2(src, IMG_OUT / src.name)

# metrics
proc = ROOT / "data" / "processed"
shutil.copy2(proc / "metrics.json", DATA_OUT / "metrics.json")
shutil.copy2(proc / "ahp_results.json", DATA_OUT / "ahp_results.json")
shutil.copy2(proc / "physical_model.json", DATA_OUT / "physical_model.json")
manifest = json.loads((proc / "layers" / "manifest.json").read_text())
(DATA_OUT / "layers_manifest.json").write_text(json.dumps(manifest))

print(f"figures: {len(list(FIG_OUT.glob('*.webp')))}  layers: {len(list(IMG_OUT.glob('*.webp')))}  data: "
      f"{', '.join(p.name for p in DATA_OUT.iterdir())}")
