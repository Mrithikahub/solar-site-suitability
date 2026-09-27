"""
Google Earth Engine feature extraction.

This module builds a single multi-band `ee.Image` ("feature stack") in which
every band is one remote-sensing predictor, then samples that stack at point
locations in batches. Sampling many points per request (instead of one request
per point) keeps extraction for ~5 000 points to a few minutes.

Sensors / products used
-----------------------
* SRTM GL1 v3 (30 m)                   -> elevation, slope, aspect
* Sentinel-2 MSI L2A (10-20 m)         -> NDVI, NDBI (Cloud Score+ masked median)
* Landsat 8/9 OLI/TIRS C2 L2 (30 m*)   -> land surface temperature (ST_B10)
* ESA WorldCover v200 (10 m, 2021)     -> land cover class
* Landsat 8 C2 L2, 2013-2015           -> pre-construction NDVI/NDBI/LST (ablation)
  (*TIRS thermal is acquired at 100 m and delivered resampled to 30 m)
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
import pandas as pd

from . import config

_EE_READY = False


# --------------------------------------------------------------------------- #
# Initialisation / region
# --------------------------------------------------------------------------- #
def init_ee() -> None:
    """Initialise Earth Engine once per process using the project in `.env`.

    If a service-account key is supplied through GEE_SERVICE_ACCOUNT_JSON
    (used for the deployed backend), that is used instead of user credentials.
    """
    global _EE_READY
    if _EE_READY:
        return
    import os

    sa_json = os.getenv("GEE_SERVICE_ACCOUNT_JSON")
    if sa_json:
        info = json.loads(sa_json)
        creds = ee.ServiceAccountCredentials(info["client_email"], key_data=sa_json)
        ee.Initialize(creds, project=config.GEE_PROJECT_ID or info.get("project_id"))
    else:
        if not config.GEE_PROJECT_ID:
            raise RuntimeError("GEE_PROJECT_ID is not set - add it to your .env file")
        ee.Initialize(project=config.GEE_PROJECT_ID)
    _EE_READY = True


def region_fc() -> ee.FeatureCollection:
    """Administrative boundary of the configured study region (FAO GAUL 2025)."""
    fc = (
        ee.FeatureCollection(config.GAUL_ASSET)
        .filter(ee.Filter.eq("GAUL0_NAME", config.REGION_COUNTRY))
        .filter(ee.Filter.eq("GAUL1_NAME", config.REGION_NAME))
    )
    return fc


def region_geometry() -> ee.Geometry:
    return region_fc().geometry()


def save_region_geojson(max_error_m: float = 150) -> dict:
    """Download a simplified copy of the region boundary for local use
    (random point sampling, grid generation, frontend mask)."""
    if config.REGION_GEOJSON.exists():
        return json.loads(config.REGION_GEOJSON.read_text())
    init_ee()
    geom = region_geometry().simplify(max_error_m)
    gj = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": config.REGION_NAME, "country": config.REGION_COUNTRY},
                "geometry": geom.getInfo(),
            }
        ],
    }
    config.REGION_GEOJSON.write_text(json.dumps(gj))
    return gj


# --------------------------------------------------------------------------- #
# Sentinel-2: NDVI and NDBI
# --------------------------------------------------------------------------- #
def sentinel2_indices(region: ee.Geometry, start: str, end: str) -> ee.Image:
    """Cloud-masked median Sentinel-2 L2A composite -> NDVI, NDBI.

    Cloud masking uses Google's Cloud Score+ product: each S2 pixel gets a
    `cs_cdf` quality score in [0, 1]; pixels below 0.60 (cloud, haze, shadow)
    are discarded before the per-pixel median is taken.

      NDVI = (NIR - Red)  / (NIR + Red)   = (B8  - B4) / (B8  + B4)
      NDBI = (SWIR1 - NIR)/ (SWIR1 + NIR) = (B11 - B8) / (B11 + B8)
    """
    cs_plus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
        .linkCollection(cs_plus, ["cs_cdf"])
    )

    def mask_and_scale(img: ee.Image) -> ee.Image:
        clear = img.select("cs_cdf").gte(0.60)
        # L2A surface reflectance is stored as integers scaled by 10 000
        return img.updateMask(clear).select(["B2", "B3", "B4", "B8", "B11", "B12"]).divide(10_000)

    median = s2.map(mask_and_scale).median()
    ndvi = median.normalizedDifference(["B8", "B4"]).rename("ndvi")
    ndbi = median.normalizedDifference(["B11", "B8"]).rename("ndbi")
    return ndvi.addBands(ndbi)


def sentinel2_true_color(region: ee.Geometry, start: str, end: str) -> ee.Image:
    """RGB median composite (for map visualisation only)."""
    cs_plus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
        .linkCollection(cs_plus, ["cs_cdf"])
        .map(lambda i: i.updateMask(i.select("cs_cdf").gte(0.60)))
    )
    return s2.select(["B4", "B3", "B2"]).median().divide(10_000)


# --------------------------------------------------------------------------- #
# Landsat 8/9: land surface temperature
# --------------------------------------------------------------------------- #
def _landsat_mask(img: ee.Image) -> ee.Image:
    """Mask fill, dilated cloud, cirrus, cloud and cloud-shadow pixels using
    the QA_PIXEL bit flags of Landsat Collection 2."""
    qa = img.select("QA_PIXEL")
    bad = (
        qa.bitwiseAnd(1 << 0)          # fill
        .Or(qa.bitwiseAnd(1 << 1))     # dilated cloud
        .Or(qa.bitwiseAnd(1 << 2))     # cirrus
        .Or(qa.bitwiseAnd(1 << 3))     # cloud
        .Or(qa.bitwiseAnd(1 << 4))     # cloud shadow
    )
    return img.updateMask(bad.eq(0))


def _landsat_scale(img: ee.Image) -> ee.Image:
    """Apply Collection-2 Level-2 scale factors.

    Surface reflectance:  SR = DN * 0.0000275 - 0.2
    Surface temperature:  T[K] = DN * 0.00341802 + 149.0  -> convert to deg C
    """
    sr = img.select("SR_B.").multiply(0.0000275).add(-0.2)
    lst = img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15).rename("lst")
    return img.addBands(sr, None, True).addBands(lst)


def landsat_lst(region: ee.Geometry, start: str, end: str) -> ee.Image:
    """Median land surface temperature (deg C) from Landsat 8 + 9 over [start, end)."""
    col = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUD_COVER", 70))
        .map(_landsat_mask)
        .map(_landsat_scale)
    )
    return col.select("lst").median().rename("lst")


def landsat_baseline(region: ee.Geometry) -> ee.Image:
    """Pre-construction (2013-2015) Landsat 8 NDVI, NDBI and LST.

    OLI band mapping: Red = SR_B4, NIR = SR_B5, SWIR1 = SR_B6.
    """
    col = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(region)
        .filterDate(config.BASELINE_START, config.BASELINE_END)
        .filter(ee.Filter.lt("CLOUD_COVER", 70))
        .map(_landsat_mask)
        .map(_landsat_scale)
    )
    med = col.select(["SR_B4", "SR_B5", "SR_B6", "lst"]).median()
    ndvi = med.normalizedDifference(["SR_B5", "SR_B4"]).rename("ndvi_pre")
    ndbi = med.normalizedDifference(["SR_B6", "SR_B5"]).rename("ndbi_pre")
    lst = med.select("lst").rename("lst_pre")
    return ndvi.addBands(ndbi).addBands(lst)


# --------------------------------------------------------------------------- #
# Terrain and land cover
# --------------------------------------------------------------------------- #
def terrain() -> ee.Image:
    """SRTM elevation (m), slope (deg) and aspect (deg clockwise from north)."""
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
    slope = ee.Terrain.slope(dem).rename("slope")
    aspect = ee.Terrain.aspect(dem).rename("aspect")
    return dem.rename("elevation").addBands(slope).addBands(aspect)


def worldcover() -> ee.Image:
    return ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("landcover")


# --------------------------------------------------------------------------- #
# Feature stack
# --------------------------------------------------------------------------- #
GEE_BANDS = [
    "elevation", "slope", "aspect",
    "ndvi", "ndbi", "lst", "landcover",
    "ndvi_pre", "ndbi_pre", "lst_pre",
]


def feature_image(region: ee.Geometry | None = None, include_baseline: bool = True) -> ee.Image:
    """All Earth Engine predictors stacked into one image."""
    init_ee()
    region = region or region_geometry()
    stack = (
        terrain()
        .addBands(sentinel2_indices(region, config.COMPOSITE_START, config.COMPOSITE_END))
        .addBands(landsat_lst(region, config.COMPOSITE_START, config.COMPOSITE_END))
        .addBands(worldcover())
    )
    if include_baseline:
        stack = stack.addBands(landsat_baseline(region))
    return stack.float()


# --------------------------------------------------------------------------- #
# Layer previews (static PNGs used by the report and as web-map overlays)
# --------------------------------------------------------------------------- #
WORLDCOVER_PALETTE = ["006400", "ffbb22", "ffff4c", "f096ff", "fa0000", "b4b4b4",
                      "f0f0f0", "0064c8", "0096a0", "00cf75", "fae6a0"]
WORLDCOVER_CODES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]

LAYER_VIS = {
    "truecolor": {"bands": ["B4", "B3", "B2"], "min": 0.0, "max": 0.28, "gamma": 1.15},
    "ndvi": {"min": -0.1, "max": 0.8,
             "palette": ["8c510a", "bf812d", "dfc27d", "f6e8c3", "c7e9c0", "74c476", "31a354", "006d2c"]},
    "ndbi": {"min": -0.45, "max": 0.2,
             "palette": ["2166ac", "67a9cf", "d1e5f0", "f7f7f7", "fddbc7", "ef8a62", "b2182b"]},
    "lst": {"min": 24, "max": 48,
            "palette": ["000004", "3b0f70", "8c2981", "de4968", "fe9f6d", "fcfdbf"]},
    "slope": {"min": 0, "max": 20,
              "palette": ["ffffcc", "c7e9b4", "7fcdbb", "41b6c4", "2c7fb8", "253494"]},
    "elevation": {"min": 0, "max": 1800,
                  "palette": ["1a9850", "91cf60", "d9ef8b", "fee08b", "fc8d59", "d73027", "ffffff"]},
    "landcover": {"min": 0, "max": 10, "palette": WORLDCOVER_PALETTE},
}


def layer_image(name: str, region: ee.Geometry) -> ee.Image:
    """Visualisation-ready single layer clipped to the region."""
    if name == "truecolor":
        img = sentinel2_true_color(region, config.COMPOSITE_START, config.COMPOSITE_END)
    elif name in ("ndvi", "ndbi"):
        img = sentinel2_indices(region, config.COMPOSITE_START, config.COMPOSITE_END).select(name)
    elif name == "lst":
        img = landsat_lst(region, config.COMPOSITE_START, config.COMPOSITE_END)
    elif name in ("slope", "elevation"):
        img = terrain().select(name)
    elif name == "landcover":
        # remap class codes to 0..10 so a simple palette can be applied
        img = worldcover().remap(WORLDCOVER_CODES, list(range(len(WORLDCOVER_CODES))))
    else:
        raise ValueError(f"unknown layer {name}")
    return img.clip(region)


def export_layer_png(name: str, out_path, width: int = 1600, tiles: int = 4) -> dict:
    """Render a layer to a PNG in Web-Mercator (EPSG:3857) so it drops straight
    onto a Leaflet map as an ImageOverlay. Returns the lat/lon bounds.

    A single request for a state-wide 12-month median exceeds Earth Engine's
    per-request memory, so the extent is split into `tiles` x `tiles` Mercator
    tiles that are rendered separately and stitched together locally.
    """
    import io
    import math

    import requests
    from PIL import Image

    init_ee()
    region = region_geometry()
    coords = region.bounds(1).coordinates().getInfo()[0]
    west, east = min(p[0] for p in coords), max(p[0] for p in coords)
    south, north = min(p[1] for p in coords), max(p[1] for p in coords)

    R = 6378137.0
    merc_y = lambda lat: R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    inv_y = lambda y: math.degrees(2 * math.atan(math.exp(y / R)) - math.pi / 2)
    x0, x1 = R * math.radians(west), R * math.radians(east)
    y0, y1 = merc_y(south), merc_y(north)
    px_per_m = width / (x1 - x0)
    height = round((y1 - y0) * px_per_m)

    xs = [x0 + (x1 - x0) * i / tiles for i in range(tiles + 1)]
    ys = [y0 + (y1 - y0) * j / tiles for j in range(tiles + 1)]
    col_px = [round((x - x0) * px_per_m) for x in xs]
    row_px = [round((y1 - y) * px_per_m) for y in ys]      # image rows grow southwards

    vis_img = layer_image(name, region).visualize(**LAYER_VIS[name])
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for i in range(tiles):
        for j in range(tiles):
            w = col_px[i + 1] - col_px[i]
            h = row_px[j] - row_px[j + 1]
            rect = ee.Geometry.Rectangle(
                [math.degrees(xs[i] / R), inv_y(ys[j]), math.degrees(xs[i + 1] / R), inv_y(ys[j + 1])],
                None, False)
            url = vis_img.getThumbURL({"region": rect, "dimensions": f"{w}x{h}", "format": "png",
                                       "crs": "EPSG:3857"})
            for attempt in range(4):
                r = requests.get(url, timeout=600)
                if r.status_code == 200:
                    break
            r.raise_for_status()
            tile = Image.open(io.BytesIO(r.content)).convert("RGBA").resize((w, h))
            canvas.paste(tile, (col_px[i], row_px[j + 1]))
    canvas.save(out_path, optimize=True)
    return {"south": south, "west": west, "north": north, "east": east}


# --------------------------------------------------------------------------- #
# Batched point sampling with on-disk cache
# --------------------------------------------------------------------------- #
def _batch_key(batch: pd.DataFrame, bands: list[str]) -> str:
    h = hashlib.sha1()
    h.update(",".join(bands).encode())
    h.update(config.COMPOSITE_START.encode() + config.COMPOSITE_END.encode())
    h.update(batch[["pid", "lat", "lon"]].round(6).to_csv(index=False).encode())
    return h.hexdigest()[:16]


def _sample_batch(image: ee.Image, batch: pd.DataFrame, bands: list[str], scale: int) -> list[dict]:
    feats = [
        ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"pid": str(r.pid)})
        for r in batch.itertuples()
    ]
    fc = ee.FeatureCollection(feats)
    # reduceRegions with `first` keeps every point, even where a band is masked
    # (e.g. no cloud-free Landsat pixel), so missing values show up as NaN.
    reduced = image.select(bands).reduceRegions(
        collection=fc,
        reducer=ee.Reducer.first().setOutputs(bands) if len(bands) == 1 else ee.Reducer.first(),
        scale=scale,
        tileScale=4,
    )
    info = reduced.getInfo()
    return [f["properties"] for f in info["features"]]


def sample_points(
    points: pd.DataFrame,
    image: ee.Image | None = None,
    bands: list[str] | None = None,
    scale: int = config.GEE_SAMPLE_SCALE_M,
    batch_size: int = config.GEE_BATCH_SIZE,
    workers: int = config.GEE_WORKERS,
    cache_tag: str = "features",
    verbose: bool = True,
) -> pd.DataFrame:
    """Sample `image` at every (lat, lon) row of `points` (needs a unique `pid`).

    Each batch result is cached as JSON under data/raw/cache/<cache_tag>/ so an
    interrupted run resumes where it stopped and re-runs cost nothing.
    """
    init_ee()
    bands = bands or GEE_BANDS
    image = image if image is not None else feature_image()
    cache_dir = config.CACHE_DIR / f"gee_{cache_tag}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    batches = [points.iloc[i : i + batch_size] for i in range(0, len(points), batch_size)]
    results: list[dict] = []
    todo = []
    for b in batches:
        path = cache_dir / f"{_batch_key(b, bands)}.json"
        if path.exists():
            results.extend(json.loads(path.read_text()))
        else:
            todo.append((b, path))
    if verbose:
        print(f"[gee] {len(batches)} batches ({len(batches) - len(todo)} cached, {len(todo)} to fetch)")

    def run(b: pd.DataFrame, path) -> list[dict]:
        import random
        import time

        last_err = None
        for attempt in range(8):
            try:
                rows = _sample_batch(image, b, bands, scale)
                path.write_text(json.dumps(rows))
                return rows
            except Exception as e:
                last_err = e
                msg = str(e)
                # Concurrency / quota throttling (HTTP 429): exponential backoff with jitter.
                if "Too Many Requests" in msg or "concurrency" in msg or "429" in msg:
                    time.sleep(min(120, 5 * 2 ** attempt) + random.uniform(0, 3))
                else:
                    time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"GEE batch failed after retries: {last_err}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run, b, p): i for i, (b, p) in enumerate(todo)}
        for n, fut in enumerate(as_completed(futs), 1):
            results.extend(fut.result())
            if verbose and (n % 5 == 0 or n == len(todo)):
                print(f"[gee]   fetched {n}/{len(todo)} batches")

    out = pd.DataFrame(results)
    for b in bands:
        if b not in out.columns:
            out[b] = float("nan")
    out["pid"] = out["pid"].astype(str)
    return out[["pid"] + bands]
