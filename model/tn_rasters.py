"""
Tamil Nadu high-resolution raster stack, rebuilt from cached data only.

The Earth Engine layer PNGs in data/processed/layers were rendered with known
visualisation parameters (min, max, palette). Each colour therefore maps back
to one value on the palette ramp, so the PNGs can be *decoded* into physical
rasters (NDVI, NDBI, LST, slope, elevation, land cover) without new Earth
Engine requests. Continuous layers are quantised to ~1/255 of their display
range (e.g. NDVI +/- 0.0035); categorical land cover decodes exactly.
Values outside the display range were clipped at render time, so decoded
values are clipped to [min, max] too (e.g. elevation > 1800 m -> 1800 m).

These rasters are combined with NASA POWER climatology (interpolated) and OSM
distance lookups to form the full Tamil Nadu predictor stack used for the AHP
and ML suitability maps.

    python -m model.tn_rasters
"""
from __future__ import annotations

import json
import math

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from . import config
from .gee import LAYER_VIS, WORLDCOVER_CODES

LAYERS_DIR = config.PROCESSED_DIR / "layers"
TN_STACK_NPZ = config.PROCESSED_DIR / "tn_raster_stack.npz"
R_EARTH = 6378137.0


def _hex_to_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)


def _palette_lut(vis: dict, n: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce Earth Engine's palette ramp: stops evenly spaced over [min, max],
    linear RGB interpolation between neighbouring stops."""
    stops = np.stack([_hex_to_rgb(c) for c in vis["palette"]])
    t = np.linspace(0, 1, n)
    pos = t * (len(stops) - 1)
    lo = np.clip(np.floor(pos).astype(int), 0, len(stops) - 2)
    frac = (pos - lo)[:, None]
    colors = stops[lo] * (1 - frac) + stops[lo + 1] * frac
    values = vis["min"] + t * (vis["max"] - vis["min"])
    return colors, values


def decode_layer(name: str, stride: int = 1) -> np.ndarray:
    """PNG -> float raster (NaN where transparent / outside the region)."""
    meta = json.loads((LAYERS_DIR / "layers.json").read_text())[name]
    rgba = np.asarray(Image.open(LAYERS_DIR / meta["file"]).convert("RGBA"))[::stride, ::stride]
    rgb = rgba[..., :3].reshape(-1, 3).astype(float)
    alpha = rgba[..., 3].reshape(-1)
    vis = LAYER_VIS[name]
    colors, values = _palette_lut(vis)
    if name == "landcover":
        # 11 discrete colours -> index -> WorldCover code
        colors = np.stack([_hex_to_rgb(c) for c in vis["palette"]])
        values = np.array(WORLDCOVER_CODES, dtype=float)
    dist, idx = cKDTree(colors).query(rgb, k=1)
    out = values[idx]
    out[alpha < 128] = np.nan
    return out.reshape(rgba.shape[:2])


def pixel_coords(name: str = "ndvi", stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Latitude of every row and longitude of every column (pixel centres) of a
    Web-Mercator layer PNG."""
    meta = json.loads((LAYERS_DIR / "layers.json").read_text())[name]
    b = meta["bounds"]
    w, h = Image.open(LAYERS_DIR / meta["file"]).size
    x0, x1 = R_EARTH * math.radians(b["west"]), R_EARTH * math.radians(b["east"])
    y1 = R_EARTH * math.log(math.tan(math.pi / 4 + math.radians(b["north"]) / 2))
    px_per_m = w / (x1 - x0)
    cols = (np.arange(0, w, stride) + stride / 2)
    rows = (np.arange(0, h, stride) + stride / 2)
    lons = np.degrees((x0 + cols / px_per_m) / R_EARTH)
    ys = y1 - rows / px_per_m
    lats = np.degrees(2 * np.arctan(np.exp(ys / R_EARTH)) - math.pi / 2)
    return lats, lons


def aspect_from_elevation(elev: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Aspect (deg clockwise from north) from the decoded DEM. The DEM is
    smoothed first because 8-bit decoding produces ~7 m terracing."""
    z = gaussian_filter(np.nan_to_num(elev, nan=np.nanmean(elev)), sigma=1.5)
    dy_m = np.gradient(lats)[:, None] * 111_320.0
    dx_m = np.gradient(lons)[None, :] * 111_320.0 * np.cos(np.deg2rad(lats))[:, None]
    dz_dy = np.gradient(z, axis=0) / dy_m      # rows go north -> south, dy_m < 0
    dz_dx = np.gradient(z, axis=1) / dx_m
    aspect = (np.degrees(np.arctan2(-dz_dx, -dz_dy)) + 360) % 360
    aspect[np.isnan(elev)] = np.nan
    return aspect


def build_stack(stride: int = 1) -> dict:
    from .osm import DistanceIndex
    from .power import PowerInterpolator

    lats, lons = pixel_coords(stride=stride)
    stack = {name: decode_layer(name, stride) for name in ("ndvi", "ndbi", "lst", "slope", "elevation", "landcover")}
    mask = ~np.isnan(stack["ndvi"])
    stack["aspect"] = aspect_from_elevation(stack["elevation"], lats, lons)

    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    flat_lat, flat_lon = LAT[mask], LON[mask]
    power = PowerInterpolator()(flat_lat, flat_lon)
    dist = DistanceIndex()(flat_lat, flat_lon)
    for col in list(power.columns) + list(dist.columns):
        arr = np.full(mask.shape, np.nan, dtype=np.float32)
        arr[mask] = (power[col] if col in power else dist[col]).to_numpy()
        stack[col] = arr
    stack = {k: v.astype(np.float32) for k, v in stack.items()}
    np.savez_compressed(TN_STACK_NPZ, lats=lats, lons=lons, mask=mask, **stack)
    # small companion file with just the pixel-centre coordinates (used by the API)
    np.savez(config.PROCESSED_DIR / "tn_grid_coords.npz", lats=lats, lons=lons)
    print(f"[tn-stack] {mask.shape[1]}x{mask.shape[0]} px, {mask.sum():,} valid -> {TN_STACK_NPZ}")
    return {"lats": lats, "lons": lons, "mask": mask, **stack}


def load_stack() -> dict:
    d = np.load(TN_STACK_NPZ)
    return {k: d[k] for k in d.files}


def validate_against_points() -> None:
    """Compare decoded raster values with Earth Engine point samples from the
    training table (nearest pixel) to quantify the decoding error."""
    import pandas as pd

    s = load_stack()
    df = pd.read_csv(config.DATASET_CSV)
    r = np.clip(np.searchsorted(-s["lats"], -df["lat"].to_numpy()), 0, len(s["lats"]) - 1)
    c = np.clip(np.searchsorted(s["lons"], df["lon"].to_numpy()), 0, len(s["lons"]) - 1)
    print("decoded-raster vs GEE point sample (pixel ~285 m vs point at 30 m):")
    for k in ("ndvi", "ndbi", "lst", "slope", "elevation"):
        vis = LAYER_VIS[k]
        a = s[k][r, c]
        b = df[k].clip(vis["min"], vis["max"]).to_numpy()
        ok = ~(np.isnan(a) | np.isnan(b))
        corr = np.corrcoef(a[ok], b[ok])[0, 1]
        print(f"  {k:10s} r = {corr:.3f}   MAE = {np.mean(np.abs(a[ok] - b[ok])):.3f}")
    lc = s["landcover"][r, c]
    ok = ~np.isnan(lc)
    print(f"  landcover  agreement = {(lc[ok] == df['landcover'].to_numpy()[ok]).mean():.1%}")


if __name__ == "__main__":
    build_stack()
    validate_against_points()
