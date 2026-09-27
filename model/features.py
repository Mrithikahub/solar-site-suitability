"""
Feature extraction + feature engineering.

`extract_features()` is the single entry point that turns a table of
(lat, lon) points into the full raw predictor set by combining three sources:

    Google Earth Engine  -> elevation, slope, aspect, NDVI, NDBI, LST, land cover
    NASA POWER           -> GHI, air temperature, cloud cover
    OpenStreetMap        -> distance to road / power line / substation

It is used identically by the training-set builder, the state-wide grid used
for the heatmap, and the live /predict API, which guarantees that the model
always sees features computed the same way.

`engineer()` then converts raw columns into the numeric design matrix used by
the ML models (circular aspect encoding, one-hot land cover, etc).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# --------------------------------------------------------------------------- #
# Column definitions
# --------------------------------------------------------------------------- #
CLIMATE_COLS = ["ghi", "temperature", "cloud_cover"]
TERRAIN_COLS = ["elevation", "slope", "aspect"]
SPECTRAL_COLS = ["ndvi", "ndbi", "lst"]
LANDCOVER_COL = "landcover"
INFRA_COLS = ["dist_road_km", "dist_powerline_km", "dist_substation_km"]
BASELINE_COLS = ["ndvi_pre", "ndbi_pre", "lst_pre"]

RAW_FEATURE_COLS = CLIMATE_COLS + TERRAIN_COLS + SPECTRAL_COLS + [LANDCOVER_COL] + INFRA_COLS + BASELINE_COLS

# Land-cover classes that actually occur in Tamil Nadu get their own dummy column.
LANDCOVER_DUMMIES = {
    10: "lc_tree", 20: "lc_shrub", 30: "lc_grass", 40: "lc_crop", 50: "lc_built",
    60: "lc_bare", 80: "lc_water", 90: "lc_wetland", 95: "lc_mangrove",
}

# Human-readable metadata used by the API / frontend / report
FEATURE_META = {
    "ghi":                {"label": "Solar irradiance (GHI)", "unit": "kWh/m²/day", "source": "NASA POWER"},
    "temperature":        {"label": "Air temperature",        "unit": "°C",         "source": "NASA POWER"},
    "cloud_cover":        {"label": "Cloud cover",            "unit": "%",          "source": "NASA POWER"},
    "elevation":          {"label": "Elevation",              "unit": "m",          "source": "SRTM 30 m"},
    "slope":              {"label": "Slope",                  "unit": "°",          "source": "SRTM 30 m"},
    "aspect":             {"label": "Aspect",                 "unit": "°",          "source": "SRTM 30 m"},
    "aspect_south":       {"label": "South-facing index",     "unit": "",           "source": "SRTM 30 m"},
    "aspect_east":        {"label": "East-facing index",      "unit": "",           "source": "SRTM 30 m"},
    "ndvi":               {"label": "NDVI",                   "unit": "",           "source": "Sentinel-2"},
    "ndbi":               {"label": "NDBI",                   "unit": "",           "source": "Sentinel-2"},
    "lst":                {"label": "Land surface temp.",     "unit": "°C",         "source": "Landsat 8/9"},
    "landcover":          {"label": "Land cover",             "unit": "",           "source": "ESA WorldCover"},
    "dist_road_km":       {"label": "Distance to road",       "unit": "km",         "source": "OpenStreetMap"},
    "dist_powerline_km":  {"label": "Distance to power line", "unit": "km",         "source": "OpenStreetMap"},
    "dist_substation_km": {"label": "Distance to substation", "unit": "km",         "source": "OpenStreetMap"},
    "ndvi_pre":           {"label": "NDVI (2013-15)",         "unit": "",           "source": "Landsat 8"},
    "ndbi_pre":           {"label": "NDBI (2013-15)",         "unit": "",           "source": "Landsat 8"},
    "lst_pre":            {"label": "LST (2013-15)",          "unit": "°C",         "source": "Landsat 8"},
}
for _code, _col in LANDCOVER_DUMMIES.items():
    FEATURE_META[_col] = {"label": f"Land cover: {config.WORLDCOVER_CLASSES[_code]}", "unit": "",
                          "source": "ESA WorldCover"}


# --------------------------------------------------------------------------- #
# Raw extraction
# --------------------------------------------------------------------------- #
_power = None
_dist = None


def _lazy_sources():
    global _power, _dist
    if _power is None:
        from .power import PowerInterpolator
        _power = PowerInterpolator()
    if _dist is None:
        from .osm import DistanceIndex
        _dist = DistanceIndex()
    return _power, _dist


def extract_features(points: pd.DataFrame, cache_tag: str = "features", include_baseline: bool = True,
                     verbose: bool = True) -> pd.DataFrame:
    """Return `points` (needs pid, lat, lon) with every raw feature column appended."""
    from . import gee

    power, dist = _lazy_sources()
    pts = points.copy()
    pts["pid"] = pts["pid"].astype(str)

    bands = gee.GEE_BANDS if include_baseline else [b for b in gee.GEE_BANDS if not b.endswith("_pre")]
    img = gee.feature_image(include_baseline=include_baseline)
    ee_df = gee.sample_points(pts[["pid", "lat", "lon"]], image=img, bands=bands, cache_tag=cache_tag,
                              verbose=verbose)
    out = pts.merge(ee_df, on="pid", how="left")

    clim = power(out["lat"].to_numpy(), out["lon"].to_numpy())
    dists = dist(out["lat"].to_numpy(), out["lon"].to_numpy())
    out = pd.concat([out.reset_index(drop=True), clim, dists], axis=1)
    return out


# --------------------------------------------------------------------------- #
# GLOBAL model features
# --------------------------------------------------------------------------- #
# MODIS MCD12Q1 IGBP (LC_Type1) classes grouped into 10 broader types
IGBP_GROUPS = {
    "lc_forest": [1, 2, 3, 4, 5], "lc_shrub": [6, 7], "lc_savanna": [8, 9], "lc_grass": [10],
    "lc_wetland": [11], "lc_crop": [12, 14], "lc_urban": [13], "lc_snow": [15],
    "lc_barren": [16], "lc_water": [17],
}
IGBP_NAMES = {
    1: "Evergreen needleleaf forest", 2: "Evergreen broadleaf forest", 3: "Deciduous needleleaf forest",
    4: "Deciduous broadleaf forest", 5: "Mixed forest", 6: "Closed shrubland", 7: "Open shrubland",
    8: "Woody savanna", 9: "Savanna", 10: "Grassland", 11: "Permanent wetland", 12: "Cropland",
    13: "Urban / built-up", 14: "Cropland / natural mosaic", 15: "Snow and ice", 16: "Barren", 17: "Water",
}
GLOBAL_EPOCH_COLS = ["ndvi", "ndbi", "lst", "landcover", "nightlights", "population"]

GLOBAL_FEATURE_META = {
    "ghi":           {"label": "Solar irradiance (GHI)", "unit": "kWh/m²/day", "source": "ERA5-Land"},
    "temperature":   {"label": "Air temperature",        "unit": "°C",         "source": "ERA5-Land"},
    "cloud_cover":   {"label": "Cloud fraction",         "unit": "%",          "source": "MODIS MOD08"},
    "elevation":     {"label": "Elevation",              "unit": "m",          "source": "Copernicus GLO-30"},
    "slope":         {"label": "Slope",                  "unit": "°",          "source": "Copernicus GLO-30"},
    "aspect":        {"label": "Aspect",                 "unit": "°",          "source": "Copernicus GLO-30"},
    "aspect_equator": {"label": "Equator-facing index",  "unit": "",           "source": "Copernicus GLO-30"},
    "aspect_east":   {"label": "East-facing index",      "unit": "",           "source": "Copernicus GLO-30"},
    "accessibility": {"label": "Travel time to city",    "unit": "min",        "source": "MAP / Oxford"},
    "log_access":    {"label": "Travel time to city (log)", "unit": "log min", "source": "MAP / Oxford"},
    "ndvi":          {"label": "NDVI",                   "unit": "",           "source": "MODIS MOD13A1"},
    "ndbi":          {"label": "NDBI",                   "unit": "",           "source": "MODIS MOD09A1"},
    "lst":           {"label": "Land surface temp.",     "unit": "°C",         "source": "MODIS MOD11A2"},
    "landcover":     {"label": "Land cover (IGBP)",      "unit": "",           "source": "MODIS MCD12Q1"},
    "nightlights":   {"label": "Night-time lights",      "unit": "nW/cm²/sr",  "source": "VIIRS DNB"},
    "log_lights":    {"label": "Night-time lights (log)", "unit": "",          "source": "VIIRS DNB"},
    "population":    {"label": "Population density",     "unit": "people/km²", "source": "GHSL"},
    "log_pop":       {"label": "Population density (log)", "unit": "",         "source": "GHSL"},
    "worldcover":    {"label": "Land cover (10 m)",      "unit": "",           "source": "ESA WorldCover"},
}
for _col, _codes in IGBP_GROUPS.items():
    GLOBAL_FEATURE_META[_col] = {"label": "Land cover: " + _col[3:].replace("_", " ").title(), "unit": "",
                                 "source": "MODIS MCD12Q1"}


def engineer_global(df: pd.DataFrame, epoch: str = "pre", landcover_source: str = "modis") -> pd.DataFrame:
    """Design matrix for the global model.

    epoch="pre"     -> time-varying predictors from the pre-construction baseline
                       (columns with suffix _pre) - used for TRAINING.
    epoch="current" -> the same predictors measured now - used for PREDICTION
                       (and for the leakage-ablation training variant).
    landcover_source="worldcover" swaps MODIS IGBP for ESA WorldCover 2021
    (only meaningful with epoch="current"; ablation only).
    """
    suffix = "_pre" if epoch == "pre" else ""
    X = pd.DataFrame(index=df.index)
    X["ghi"] = df["ghi"].astype(float)
    X["temperature"] = df["temperature"].astype(float)
    X["cloud_cover"] = df["cloud_cover"].astype(float)
    X["elevation"] = df["elevation"].astype(float)
    X["slope"] = df["slope"].astype(float)

    # Equator-facing slopes receive more irradiance: south in the northern
    # hemisphere, north in the southern. Weighted by steepness (0 when flat).
    asp = np.deg2rad(df["aspect"].astype(float))
    steep = np.clip(df["slope"].astype(float) / 10.0, 0, 1)
    hemi = np.where(df["lat"].astype(float) >= 0, 1.0, -1.0)
    X["aspect_equator"] = (-np.cos(asp) * hemi * steep).fillna(0)
    X["aspect_east"] = (np.sin(asp) * steep).fillna(0)

    X["log_access"] = np.log1p(df["accessibility"].astype(float).clip(lower=0))
    X["ndvi"] = df[f"ndvi{suffix}"].astype(float)
    X["ndbi"] = df[f"ndbi{suffix}"].astype(float)
    X["lst"] = df[f"lst{suffix}"].astype(float)
    X["log_lights"] = np.log1p(df[f"nightlights{suffix}"].astype(float).clip(lower=0))
    X["log_pop"] = np.log1p(df[f"population{suffix}"].astype(float).clip(lower=0))

    if landcover_source == "worldcover":
        wc = df["worldcover"].round()
        for code, col in LANDCOVER_DUMMIES.items():
            X[col] = (wc == code).astype(int)
    else:
        lc = df[f"landcover{suffix}"].round()
        for col, codes in IGBP_GROUPS.items():
            X[col] = lc.isin(codes).astype(int)
    return X


# --------------------------------------------------------------------------- #
# Feature engineering for ML (Tamil Nadu local model)
# --------------------------------------------------------------------------- #
def engineer(df: pd.DataFrame, include_spectral: bool = True, use_baseline: bool = False,
             include_landcover: bool = True) -> pd.DataFrame:
    """Build the model design matrix from raw feature columns.

    * aspect (circular, 0 = 360) -> aspect_south = -cos(aspect), aspect_east = sin(aspect).
      Weighted by slope-steepness so aspect only matters where terrain is not flat.
    * land cover -> one-hot dummies (tree, crop, bare ...)
    * use_baseline=True swaps current NDVI/NDBI/LST for their 2013-15 Landsat
      values (the label-leakage control). ESA WorldCover only exists for
      2020/2021 - after the plants were built - so the clean model is trained
      with include_landcover=False.
    """
    X = pd.DataFrame(index=df.index)
    for c in CLIMATE_COLS + ["elevation", "slope"]:
        X[c] = df[c].astype(float)

    asp = np.deg2rad(df["aspect"].astype(float))
    steep = np.clip(df["slope"].astype(float) / 10.0, 0, 1)   # 0 on flat ground, 1 above 10 deg
    X["aspect_south"] = (-np.cos(asp) * steep).fillna(0)
    X["aspect_east"] = (np.sin(asp) * steep).fillna(0)

    if include_spectral:
        src = BASELINE_COLS if use_baseline else SPECTRAL_COLS
        for s, c in zip(src, SPECTRAL_COLS):
            X[c] = df[s].astype(float)
        if include_landcover:
            lc = df[LANDCOVER_COL].round()
            for code, col in LANDCOVER_DUMMIES.items():
                X[col] = (lc == code).astype(int)

    for c in INFRA_COLS:
        X[c] = df[c].astype(float)
    return X
