"""
Phase 1 - build the labelled training dataset.

    python -m model.build_dataset

Steps
  1. Region boundary (FAO GAUL 2025) from Earth Engine -> data/raw/region_boundary.geojson
  2. Solar sites from OSM -> positive sample points inside each footprint
  3. Negative points: uniform random over the region, excluding a buffer
     around every solar site
  4. NASA POWER lattice + OSM infrastructure index (cached)
  5. Feature extraction for all points -> data/processed/dataset.csv
"""
from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely.ops import unary_union

from . import config, gee, osm, power
from .features import RAW_FEATURE_COLS, extract_features


def sample_positives(sites: gpd.GeoDataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Random points inside each solar site; count ~ area, capped per site."""
    m = sites.to_crs(config.METRIC_EPSG)
    rows = []
    for site, geom in zip(m.itertuples(), m.geometry):
        area_ha = geom.area / 10_000
        n = int(np.clip(round(area_ha * config.POSITIVE_POINTS_PER_HA), 1, config.POSITIVE_MAX_PER_POLYGON))
        inner = geom.buffer(-config.POSITIVE_INNER_BUFFER_M)
        if inner.is_empty or inner.area < 100:
            pts = [geom.representative_point()]
        else:
            pts = list(gpd.GeoSeries([inner], crs=config.METRIC_EPSG)
                       .sample_points(n, rng=rng).explode(index_parts=False))
        for p in pts:
            rows.append({"site_id": site.site_id, "x": p.x, "y": p.y, "site_area_ha": area_ha})
    df = pd.DataFrame(rows)
    lon, lat = Transformer.from_crs(config.METRIC_EPSG, 4326, always_xy=True).transform(df["x"], df["y"])
    df["lat"], df["lon"] = lat, lon
    df["label"] = 1
    df["group_id"] = df["site_id"]
    return df.drop(columns=["x", "y"])


def sample_negatives(region, sites: gpd.GeoDataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Uniform random points in the region, at least NEGATIVE_EXCLUSION_BUFFER_M
    from any solar site. Grouped into 0.25 deg blocks for spatial CV."""
    excl = unary_union(sites.to_crs(config.METRIC_EPSG).geometry.buffer(config.NEGATIVE_EXCLUSION_BUFFER_M))
    shapely.prepare(region)
    shapely.prepare(excl)
    to_m = Transformer.from_crs(4326, config.METRIC_EPSG, always_xy=True)
    minx, miny, maxx, maxy = region.bounds
    out = []
    while sum(len(o) for o in out) < n:
        lon = rng.uniform(minx, maxx, n * 3)
        lat = rng.uniform(miny, maxy, n * 3)
        inside = shapely.contains_xy(region, lon, lat)
        lon, lat = lon[inside], lat[inside]
        x, y = to_m.transform(lon, lat)
        keep = ~shapely.contains_xy(excl, x, y)
        out.append(pd.DataFrame({"lat": lat[keep], "lon": lon[keep]}))
    df = pd.concat(out).iloc[:n].reset_index(drop=True)
    df["label"] = 0
    df["site_id"] = ""
    df["site_area_ha"] = np.nan
    df["group_id"] = ("B" + (df["lat"] // 0.25).astype(int).astype(str) + "_"
                      + (df["lon"] // 0.25).astype(int).astype(str))
    return df


def main() -> pd.DataFrame:
    rng = np.random.default_rng(config.RANDOM_SEED)
    gee.init_ee()

    # 1. region ----------------------------------------------------------------
    gj = gee.save_region_geojson()
    region = osm.region_polygon_from_geojson(gj)
    print(f"[region] {config.REGION_NAME}: bounds {tuple(round(b, 3) for b in region.bounds)}")

    # 2. solar sites -> positives ----------------------------------------------
    sites = osm.fetch_solar_sites(region)
    sites.to_file(config.SOLAR_POLYGONS_GEOJSON, driver="GeoJSON")
    print(f"[labels] {len(sites)} solar sites >= {config.MIN_SOLAR_AREA_M2} m2, "
          f"total {sites['area_m2'].sum() / 1e6:.1f} km2")
    pos = sample_positives(sites, rng)
    print(f"[labels] {len(pos)} positive points")

    # 3. negatives ---------------------------------------------------------------
    n_neg = int(len(pos) * config.NEGATIVE_TO_POSITIVE_RATIO)
    neg = sample_negatives(region, sites, n_neg, rng)
    print(f"[labels] {len(neg)} negative points")

    pts = pd.concat([pos, neg], ignore_index=True)
    pts["pid"] = [f"P{i:05d}" for i in range(len(pts))]

    # 4. auxiliary sources ---------------------------------------------------------
    if not config.POWER_GRID_CSV.exists():
        power.build_power_grid(region.bounds)
    if not config.ROADS_KDTREE.exists():
        osm.build_infrastructure_index(region.bounds)

    # 5. features ----------------------------------------------------------------
    df = extract_features(pts, cache_tag="training")
    cols = ["pid", "lat", "lon", "label", "group_id", "site_id", "site_area_ha"] + RAW_FEATURE_COLS
    df = df[cols]
    df.to_csv(config.DATASET_CSV, index=False)
    print(f"[done] {len(df)} rows -> {config.DATASET_CSV}")
    return df


if __name__ == "__main__":
    main()
