"""
Export every web-map overlay + a manifest the API/frontend reads.

Global layers are rendered from the cached 0.5 deg feature grid (no Earth
Engine calls); Tamil Nadu layers are the Earth Engine PNGs rendered in
Phase 1 plus the AHP / ML suitability rasters. Every overlay is a
Web-Mercator PNG with lat/lon bounds so Leaflet can drop it in as an
ImageOverlay.

    python -m model.export_layers
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import config, viz
from .gee import LAYER_VIS, WORLDCOVER_CODES, WORLDCOVER_PALETTE
from .global_grid import grid_features_path

LAYERS_DIR = config.PROCESSED_DIR / "layers"
MANIFEST = LAYERS_DIR / "manifest.json"

GHI_STEPS = ["#fff3d6", "#fbd98a", "#f5bd3c", "#eda100", "#bf8000", "#8a5a00"]
LIGHTS_STEPS = ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#ffffff"]
POP_STEPS = ["#f3eefb", "#cfc3f0", "#9f8be0", "#6f56c7", "#4a3aa7", "#2b1f73"]


def _cmap(steps):
    return LinearSegmentedColormap.from_list("c", steps)


def _hexs(vis):
    return ["#" + c.lstrip("#") for c in vis["palette"]]


GLOBAL_LAYERS = {
    "ghi": {"title": "Solar irradiance (GHI)", "unit": "kWh/m²/day", "source": "ERA5-Land 2005-2024",
            "min": 2.0, "max": 7.0, "palette": GHI_STEPS},
    "ndvi": {"title": "NDVI", "unit": "", "source": "MODIS MOD13A1 (last 12 months)",
             "min": LAYER_VIS["ndvi"]["min"], "max": LAYER_VIS["ndvi"]["max"], "palette": _hexs(LAYER_VIS["ndvi"])},
    "slope": {"title": "Slope", "unit": "°", "source": "Copernicus GLO-30 DEM",
              "min": 0, "max": 20, "palette": _hexs(LAYER_VIS["slope"])},
    "lst": {"title": "Land surface temperature", "unit": "°C", "source": "MODIS MOD11A2 (day)",
            "min": -10, "max": 45, "palette": _hexs(LAYER_VIS["lst"])},
    "nightlights": {"title": "Night-time lights (log)", "unit": "log(1+nW/cm²/sr)", "source": "VIIRS DNB monthly",
                    "min": 0, "max": 4, "palette": LIGHTS_STEPS, "transform": "log1p"},
    "population": {"title": "Population density (log)", "unit": "log(1+people/km²)", "source": "GHSL 2025",
                   "min": 0, "max": 8, "palette": POP_STEPS, "transform": "log1p"},
}


def export_global() -> dict:
    g = pd.read_csv(grid_features_path(0.5))
    step, lat0, lon0 = 0.5, config.GLOBAL_LAT_RANGE[0], -180.0
    nrows = int((config.GLOBAL_LAT_RANGE[1] - lat0) / step)
    out = {}
    for name, spec in GLOBAL_LAYERS.items():
        v = g[name].to_numpy(float)
        if spec.get("transform") == "log1p":
            v = np.log1p(np.clip(v, 0, None))
        grid = viz.points_to_grid(g["lat"].to_numpy(), g["lon"].to_numpy(), v, lat0, lon0, step, nrows, 720)
        merc, bounds = viz.grid_to_mercator(grid, lat0, lon0, step, width=2048)
        rgba = viz.colorize(merc, cmap=_cmap(spec["palette"]), vmin=spec["min"], vmax=spec["max"],
                            alpha_ramp=(0.85, 0.85))
        fname = f"global_{name}.png"
        viz.save_png(rgba, LAYERS_DIR / fname)
        out[name] = {**spec, "file": fname, "bounds": bounds, "kind": "continuous"}

    # land cover: ESA WorldCover class at each cell centre (categorical)
    wc_colors = {c: "#" + p for c, p in zip(WORLDCOVER_CODES, WORLDCOVER_PALETTE)}
    grid = viz.points_to_grid(g["lat"].to_numpy(), g["lon"].to_numpy(), g["worldcover"].to_numpy(float),
                              lat0, lon0, step, nrows, 720)
    merc, bounds = viz.grid_to_mercator(grid, lat0, lon0, step, width=2048, categorical=True)
    viz.save_png(viz.categorical_rgba(np.round(np.nan_to_num(merc, nan=-1)), wc_colors), LAYERS_DIR / "global_landcover.png")
    out["landcover"] = {"title": "Land cover (ESA WorldCover)", "unit": "", "source": "ESA WorldCover 2021 (cell centre)",
                        "file": "global_landcover.png", "bounds": bounds, "kind": "categorical",
                        "classes": [{"code": c, "name": config.WORLDCOVER_CLASSES[c], "color": wc_colors[c]}
                                    for c in WORLDCOVER_CODES if c != 70]}
    for key, fname, title, desc in (
            ("suitability", "physical_global.png", "Suitability",
             "Physical suitability: sunshine, cloud, slope, vegetation and land cover (region-balanced PU model)."),
            ("development", "ml_global.png", "Development likelihood",
             "Resemblance to where plants have historically been built - near infrastructure and demand."),
            ("ahp", "ahp_global.png", "AHP score", "Expert-weighted GIS overlay baseline.")):
        out[key] = {"title": title, "description": desc, "unit": "0-100", "source": "This project (0.5° grid)",
                    "file": fname, "bounds": bounds, "kind": "continuous", "min": 0, "max": 100,
                    "palette": viz.SUIT_STEPS}
    return out


def export_tn() -> dict:
    from .tn_rasters import load_stack

    base = json.loads((LAYERS_DIR / "layers.json").read_text())
    bounds = base["ndvi"]["bounds"]
    titles = {"truecolor": "Sentinel-2 true colour", "ndvi": "NDVI (Sentinel-2)", "ndbi": "NDBI (Sentinel-2)",
              "lst": "Land surface temperature (Landsat 8/9)", "slope": "Slope (SRTM)",
              "elevation": "Elevation (SRTM)", "landcover": "Land cover (ESA WorldCover 10 m)"}
    units = {"ndvi": "", "ndbi": "", "lst": "°C", "slope": "°", "elevation": "m", "truecolor": "", "landcover": ""}
    sources = {"truecolor": "Sentinel-2 L2A B4-B3-B2", "ndvi": "Sentinel-2 L2A", "ndbi": "Sentinel-2 L2A",
               "lst": "Landsat 8/9 C2 L2 ST_B10", "slope": "SRTM GL1 30 m", "elevation": "SRTM GL1 30 m",
               "landcover": "ESA WorldCover v200"}
    out = {}
    for name, meta in base.items():
        vis = meta["vis"]
        entry = {"title": titles[name], "unit": units[name], "source": sources[name] + " · Sep 2025 - Sep 2026",
                 "file": meta["file"], "bounds": bounds}
        if name == "landcover":
            entry.update(kind="categorical", classes=[
                {"code": c, "name": config.WORLDCOVER_CLASSES[c], "color": "#" + p}
                for c, p in zip(WORLDCOVER_CODES, WORLDCOVER_PALETTE) if c not in (70, 100)])
        elif name == "truecolor":
            entry.update(kind="image")
        else:
            entry.update(kind="continuous", min=vis["min"], max=vis["max"], palette=_hexs(vis))
        out[name] = entry

    # GHI from NASA POWER (interpolated on the TN raster frame)
    s = load_stack()
    ghi = s["ghi"].astype(float)
    ghi[~s["mask"]] = np.nan
    lo, hi = float(np.nanpercentile(ghi, 1)), float(np.nanpercentile(ghi, 99))
    viz.save_png(viz.colorize(ghi, cmap=_cmap(GHI_STEPS), vmin=lo, vmax=hi, alpha_ramp=(0.85, 0.85)),
                 LAYERS_DIR / "tn_ghi.png")
    out["ghi"] = {"title": "Solar irradiance (GHI)", "unit": "kWh/m²/day", "source": "NASA POWER 2005-2024",
                  "file": "tn_ghi.png", "bounds": bounds, "kind": "continuous", "min": round(lo, 2),
                  "max": round(hi, 2), "palette": GHI_STEPS}
    for key, fname, title in (("suitability", "ml_tn.png", "ML suitability (local model)"),
                              ("ahp", "ahp_tn.png", "AHP suitability")):
        out[key] = {"title": title, "unit": "0-100", "source": "This project (~285 m)", "file": fname,
                    "bounds": bounds, "kind": "continuous", "min": 0, "max": 100, "palette": viz.SUIT_STEPS}
    return out


def to_webp() -> None:
    """Lossy WebP copies (alpha kept) - 5-10x smaller than PNG for the web map."""
    from PIL import Image

    for png in LAYERS_DIR.glob("*.png"):
        Image.open(png).save(png.with_suffix(".webp"), "WEBP", quality=82, method=6)


def main():
    manifest = {"global": export_global(), "tamil_nadu": export_tn()}
    to_webp()
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    for scope, layers in manifest.items():
        print(f"[layers] {scope}: {', '.join(layers)}")


if __name__ == "__main__":
    main()
