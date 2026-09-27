"""
Central configuration for the Solar Site Suitability project.

Everything that describes *where* and *when* we look (study region, time windows,
sampling sizes, file locations) lives here, so the whole pipeline can be pointed
at a different state/country by editing `.env` rather than code.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = RAW_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "model"
REPORT_DIR = ROOT_DIR / "report"
FIGURES_DIR = REPORT_DIR / "figures"

for _d in (RAW_DIR, CACHE_DIR, PROCESSED_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATASET_CSV = PROCESSED_DIR / "dataset.csv"
REGION_GEOJSON = RAW_DIR / "region_boundary.geojson"
SOLAR_POLYGONS_GEOJSON = RAW_DIR / "osm_solar_polygons.geojson"
POWER_GRID_CSV = RAW_DIR / "nasa_power_grid.csv"
ROADS_KDTREE = RAW_DIR / "osm_roads_vertices.npy"
POWERLINES_KDTREE = RAW_DIR / "osm_powerlines_vertices.npy"
SUBSTATIONS_NPY = RAW_DIR / "osm_substations.npy"

# --------------------------------------------------------------------------- #
# Google Earth Engine
# --------------------------------------------------------------------------- #
GEE_PROJECT_ID = os.getenv("GEE_PROJECT_ID", "")

# --------------------------------------------------------------------------- #
# Study region (configurable through .env)
# --------------------------------------------------------------------------- #
REGION_NAME = os.getenv("REGION_NAME", "Tamil Nadu")
REGION_COUNTRY = os.getenv("REGION_COUNTRY", "India")
# FAO GAUL 2025 level-1 administrative boundaries
GAUL_ASSET = "FAO/GAUL/2025/level1"

# Metric CRS used for all distance / area computations.
# UTM zone 44N covers Tamil Nadu; change together with REGION_NAME.
METRIC_EPSG = int(os.getenv("METRIC_EPSG", "32644"))

# --------------------------------------------------------------------------- #
# Temporal windows for optical / thermal composites
# --------------------------------------------------------------------------- #
# "Current" land-surface state: last 12 months (fixed dates for reproducibility).
COMPOSITE_START = os.getenv("COMPOSITE_START", "2025-09-01")
COMPOSITE_END = os.getenv("COMPOSITE_END", "2026-09-01")

# "Pre-construction" baseline, used for the label-leakage ablation study:
# most Tamil Nadu solar parks were commissioned 2015-2020, so Landsat 8 scenes
# from 2013-2014 show the land *before* panels were installed.
BASELINE_START = "2013-04-01"
BASELINE_END = "2015-03-31"

# NASA POWER climatology period (long-term mean).
POWER_START_YEAR = 2005
POWER_END_YEAR = 2024
POWER_GRID_STEP_DEG = 0.5

# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42
GEE_SAMPLE_SCALE_M = 30          # common analysis resolution (Landsat / SRTM native)
GEE_BATCH_SIZE = 400             # points per sampleRegions request
GEE_WORKERS = 3                  # concurrent Earth Engine requests (project is in Restricted Mode)

MIN_SOLAR_AREA_M2 = 5_000        # ignore rooftop-sized generator polygons
POSITIVE_INNER_BUFFER_M = 20     # keep positive samples away from polygon edges
POSITIVE_POINTS_PER_HA = 0.25    # 1 sample per 4 ha of plant area ...
POSITIVE_MAX_PER_POLYGON = 20    # ... capped so mega-parks don't dominate
NEGATIVE_EXCLUSION_BUFFER_M = 1_000
NEGATIVE_TO_POSITIVE_RATIO = 1.5

# Overpass mirrors, tried in order (the main server is frequently overloaded).
OVERPASS_URLS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
HTTP_USER_AGENT = "solarsite/1.0 (+https://github.com/Mrithikahub/solar-site-suitability)"

# --------------------------------------------------------------------------- #
# GLOBAL model
# --------------------------------------------------------------------------- #
GLOBAL_DATASET_CSV = PROCESSED_DIR / "global_dataset.csv"
TN_GLOBAL_FEATURES_CSV = PROCESSED_DIR / "tn_global_features.csv"
WORLD_REGIONS_GEOJSON = RAW_DIR / "world_regions_lsib.geojson"
GLOBAL_PV_ASSET = "projects/sat-io/open-datasets/global_photovoltaic/predicted_set"
GLOBAL_PV_CSV = RAW_DIR / "global_pv_plants.csv"

# Pre-construction epoch for time-varying predictors (Kruitwagen plants were
# built up to 2018; most after 2010) and "current" epoch used at prediction time.
GLOBAL_BASELINE = {
    "composite": ("2008-01-01", "2010-01-01"),   # MODIS NDVI / NDBI / LST
    "landcover_year": 2008,                       # MODIS MCD12Q1
    "population_epoch": "2010",                   # GHSL
    "lights": ("2014-01-01", "2015-01-01"),       # earliest full VIIRS year
}
GLOBAL_CURRENT = {
    "composite": (COMPOSITE_START, COMPOSITE_END),
    "landcover_year": 2024,
    "population_epoch": "2025",
    "lights": (COMPOSITE_START, COMPOSITE_END),
}
CLIMATOLOGY_YEARS = ("2005-01-01", "2025-01-01")  # ERA5-Land / MODIS cloud climatology

GLOBAL_MIN_PLANT_AREA_M2 = 10_000
GLOBAL_MAX_POS_PER_REGION = 1_400
GLOBAL_N_NEG_BACKGROUND = 4_500
GLOBAL_N_NEG_HARD = 4_500
GLOBAL_HARD_NEG_KM = (5, 100)
GLOBAL_LAT_RANGE = (-56.0, 72.0)                 # sampling latitudes (no Antarctica)
GLOBAL_GRID_DEG = 0.25

# LSIB 'wld_rgn' -> coarser reporting region
REGION_GROUPS = {
    "Europe": "Europe",
    "North America": "North America", "Central America": "Latin America",
    "Caribbean": "Latin America", "South America": "Latin America",
    "Africa": "Africa",
    "SW Asia": "Middle East", "Central Asia": "Central & North Asia", "N Asia": "Central & North Asia",
    "S Asia": "South Asia", "E Asia": "East Asia", "SE Asia": "Southeast Asia",
    "Australia": "Oceania", "Oceania": "Oceania",
    "Indian Ocean": "Africa", "S Atlantic": "Latin America", "Antarctica": "Antarctica",
}

# ESA WorldCover v200 class codes -> names
WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}
