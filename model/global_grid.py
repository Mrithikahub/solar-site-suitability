"""
Global prediction grid (0.5 deg by default, 0.25 deg for the full-resolution run).

Rather than exporting coarse aggregated rasters (which would, for example,
flatten 30 m slopes into ~0 deg at 27 km resolution and bias the model), we
sample the *same* current-epoch feature stack used by the live API at the
centre of every land cell. This keeps the heatmap statistically consistent
with the training data. Points are sampled in cached batches through Earth
Engine's reduceRegions, so an interrupted run resumes where it stopped.

    python -m model.global_grid            # 0.5 deg  (~55 000 cells)
    python -m model.global_grid 0.25       # 0.25 deg (~210 000 cells)
"""
from __future__ import annotations

import sys

import geopandas as gpd
import numpy as np
import pandas as pd

from . import config, gee, gee_global
from .build_global_dataset import world_regions


def grid_features_path(step: float):
    return config.PROCESSED_DIR / f"global_grid_features_{str(step).replace('.', 'p')}.csv.gz"


def land_cell_centres(step: float) -> pd.DataFrame:
    """Cell centres that fall on land, found with a spatial-index join against
    the individual LSIB country polygons (much faster than testing millions of
    points against one merged world polygon)."""
    regions = world_regions()
    regions = regions[regions["region"] != "Antarctica"][["country_na", "region", "geometry"]]
    lo_lat, hi_lat = config.GLOBAL_LAT_RANGE
    lats = np.arange(lo_lat + step / 2, hi_lat, step)
    lons = np.arange(-180 + step / 2, 180, step)
    LON, LAT = np.meshgrid(lons, lats)
    pts = gpd.GeoDataFrame({"lat": np.round(LAT.ravel(), 4), "lon": np.round(LON.ravel(), 4)},
                           geometry=gpd.points_from_xy(LON.ravel(), LAT.ravel()), crs=4326)
    j = gpd.sjoin(pts, regions, how="inner", predicate="within")
    j = j[~j.index.duplicated(keep="first")].sort_index()
    df = pd.DataFrame(j[["lat", "lon", "country_na", "region"]]).reset_index(drop=True)
    df["pid"] = [f"C{str(step).replace('.', '')}_{i:06d}" for i in range(len(df))]
    return df


def build_grid_features(step: float = 0.5) -> pd.DataFrame:
    cells = land_cell_centres(step)
    print(f"[grid] {len(cells):,} land cells at {step} deg", flush=True)
    feats = gee.sample_points(cells[["pid", "lat", "lon"]], image=gee_global.prediction_image(),
                              bands=gee_global.PREDICTION_BANDS, cache_tag=f"global_grid_{step}",
                              batch_size=500, workers=config.GEE_WORKERS)
    df = cells.merge(feats, on="pid", how="left")
    out = grid_features_path(step)
    df.to_csv(out, index=False, compression="gzip")
    print(f"[grid] saved -> {out}", flush=True)
    return df


if __name__ == "__main__":
    build_grid_features(float(sys.argv[1]) if len(sys.argv) > 1 else 0.5)
