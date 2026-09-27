"""
Phase 1b - build the GLOBAL labelled dataset.

    python -m model.build_global_dataset

Labels
  positives : Kruitwagen et al. (2021, Nature) global PV inventory - 68 661
              plant polygons detected in Sentinel-2 / SPOT imagery (2016-2018).
              One random interior point per plant >= 1 ha, stratified by world
              region (capped) so no single region dominates.
  negatives : (a) background - area-uniform random points on land
              (b) hard       - 5-100 km from a random plant (same landscape
                               context, but no PV), which forces the model to
                               learn *local* suitability, not just "sunny and
                               populated".
              All negatives are >= 1 km + plant radius from every plant.

Every point is tagged with a world region (US State Dept. LSIB) for
stratified reporting and spatial cross-validation.
"""
from __future__ import annotations

import json

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import Point

from . import config, gee, gee_global

EARTH_R_KM = 6371.0088


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _unit_xyz(lat, lon) -> np.ndarray:
    """lat/lon (deg) -> unit vectors, so chord distances in a KD-tree are
    monotonic with great-circle distance (works across the dateline/poles)."""
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def _chord_to_km(chord):
    return 2 * EARTH_R_KM * np.arcsin(np.clip(chord / 2, 0, 1))


def _destination(lat, lon, bearing_deg, dist_km):
    """Great-circle destination point."""
    la1, lo1, b = np.deg2rad(lat), np.deg2rad(lon), np.deg2rad(bearing_deg)
    d = dist_km / EARTH_R_KM
    la2 = np.arcsin(np.sin(la1) * np.cos(d) + np.cos(la1) * np.sin(d) * np.cos(b))
    lo2 = lo1 + np.arctan2(np.sin(b) * np.sin(d) * np.cos(la1), np.cos(d) - np.sin(la1) * np.sin(la2))
    return np.rad2deg(la2), (np.rad2deg(lo2) + 540) % 360 - 180


# --------------------------------------------------------------------------- #
# Reference data
# --------------------------------------------------------------------------- #
def world_regions() -> gpd.GeoDataFrame:
    """LSIB country polygons (simplified to ~2 km) with world region."""
    if config.WORLD_REGIONS_GEOJSON.exists():
        return gpd.read_file(config.WORLD_REGIONS_GEOJSON)
    gee.init_ee()
    fc = (ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
          .map(lambda f: f.simplify(2000).select(["country_na", "wld_rgn"])))
    gdf = ee.data.computeFeatures({"expression": fc, "fileFormat": "GEOPANDAS_GEODATAFRAME"})
    gdf = gdf.set_crs(4326, allow_override=True)
    gdf["geometry"] = gdf.geometry.make_valid()
    gdf["region"] = gdf["wld_rgn"].map(config.REGION_GROUPS).fillna("Other")
    gdf.to_file(config.WORLD_REGIONS_GEOJSON, driver="GeoJSON")
    return gdf


def pv_plants() -> pd.DataFrame:
    """Attributes + centroid of every plant in the global PV inventory (cached)."""
    if config.GLOBAL_PV_CSV.exists():
        return pd.read_csv(config.GLOBAL_PV_CSV)
    gee.init_ee()
    fc = ee.FeatureCollection(config.GLOBAL_PV_ASSET)

    def attrs(f):
        c = f.geometry().centroid(10).coordinates()
        return ee.Feature(None, {
            "unique_id": f.get("unique_id"), "area_m2": f.get("area"), "capacity_mw": f.get("capacity_m"),
            "install_date": f.get("install_da"), "iso2": f.get("iso-3166-1"), "confidence": f.get("confidence"),
            "lc_before": f.get("lc_vis"), "lon": c.get(0), "lat": c.get(1),
        })
    df = ee.data.computeFeatures({"expression": fc.map(attrs), "fileFormat": "PANDAS_DATAFRAME"})
    df = df.drop(columns=[c for c in df.columns if c == "geo"], errors="ignore")
    df.to_csv(config.GLOBAL_PV_CSV, index=False)
    return df


def plant_interior_points(ids: list[int], rng: np.random.Generator) -> pd.DataFrame:
    """Download the selected plant polygons (in chunks) and draw one random
    point inside each, >= 10 m from its edge where the shape allows it."""
    gee.init_ee()
    cache = config.CACHE_DIR / "global_pv_interior_points.csv"
    if cache.exists():
        return pd.read_csv(cache)
    fc = ee.FeatureCollection(config.GLOBAL_PV_ASSET)
    rows = []
    for i in range(0, len(ids), 2000):
        chunk = ids[i:i + 2000]
        sub = fc.filter(ee.Filter.inList("unique_id", chunk)).map(
            lambda f: f.simplify(5).select(["unique_id"]))
        gdf = ee.data.computeFeatures({"expression": sub, "fileFormat": "GEOPANDAS_GEODATAFRAME"})
        gdf = gdf.set_crs(4326, allow_override=True)
        for uid, geom in zip(gdf["unique_id"], gdf.geometry):
            # local azimuthal-equidistant projection => metre buffers anywhere on Earth
            c = geom.centroid
            aeqd = f"+proj=aeqd +lat_0={c.y} +lon_0={c.x} +units=m"
            g = gpd.GeoSeries([geom], crs=4326).to_crs(aeqd)
            inner = g.buffer(-10)
            target = inner if (not inner.iloc[0].is_empty and inner.iloc[0].area > 50) else g
            p = target.sample_points(1, rng=rng).to_crs(4326).iloc[0]
            if p is not None and p.geom_type == "MultiPoint":
                p = p.geoms[0] if len(p.geoms) else None
            if p is None or p.is_empty:
                p = geom.representative_point()
            rows.append({"unique_id": uid, "lat": p.y, "lon": p.x})
        print(f"[pv] interior points {min(i + 2000, len(ids))}/{len(ids)}")
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False)
    return out


def assign_region(df: pd.DataFrame, regions: gpd.GeoDataFrame) -> pd.DataFrame:
    pts = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326)
    j = gpd.sjoin(pts, regions[["country_na", "region", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    # points on coastlines just outside the simplified polygons -> nearest country
    miss = j["region"].isna()
    if miss.any():
        near = gpd.sjoin_nearest(pts[miss].to_crs(3857), regions[["country_na", "region", "geometry"]].to_crs(3857),
                                 how="left")
        near = near[~near.index.duplicated(keep="first")]
        j.loc[miss, "country_na"] = near["country_na"].values
        j.loc[miss, "region"] = near["region"].values
    return pd.DataFrame(j.drop(columns=["geometry", "index_right"], errors="ignore"))


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def select_positives(plants: pd.DataFrame, regions: gpd.GeoDataFrame, rng) -> pd.DataFrame:
    p = plants[(plants["area_m2"] >= config.GLOBAL_MIN_PLANT_AREA_M2) & (plants["confidence"].isin(["A", "B", "C"]))]
    p = assign_region(p, regions)
    parts = []
    for reg, sub in p.groupby("region"):
        n = min(len(sub), config.GLOBAL_MAX_POS_PER_REGION)
        # area-weighted without replacement: favour utility-scale plants
        w = np.sqrt(sub["area_m2"].to_numpy())
        idx = rng.choice(len(sub), size=n, replace=False, p=w / w.sum())
        parts.append(sub.iloc[idx])
    sel = pd.concat(parts)
    print("[pv] positives per region:\n" + sel["region"].value_counts().to_string())
    return sel


def land_filter(lat, lon, land_union) -> np.ndarray:
    return shapely.contains_xy(land_union, lon, lat)


def sample_negatives(plants: pd.DataFrame, positives: pd.DataFrame, regions: gpd.GeoDataFrame, rng) -> pd.DataFrame:
    land = shapely.union_all(regions[regions["region"] != "Antarctica"].geometry.values)
    shapely.prepare(land)

    tree = cKDTree(_unit_xyz(plants["lat"].to_numpy(), plants["lon"].to_numpy()))
    radius_km = np.sqrt(plants["area_m2"].to_numpy() / np.pi) / 1000.0

    def far_from_plants(lat, lon):
        d, i = tree.query(_unit_xyz(lat, lon), k=1)
        return _chord_to_km(d) > (1.0 + radius_km[i])

    # (a) background: area-uniform over the sphere within the latitude band
    lo_lat, hi_lat = config.GLOBAL_LAT_RANGE
    s0, s1 = np.sin(np.deg2rad(lo_lat)), np.sin(np.deg2rad(hi_lat))
    bg = []
    while sum(len(b) for b in bg) < config.GLOBAL_N_NEG_BACKGROUND:
        n = 20_000
        lat = np.rad2deg(np.arcsin(rng.uniform(s0, s1, n)))
        lon = rng.uniform(-180, 180, n)
        ok = land_filter(lat, lon, land)
        lat, lon = lat[ok], lon[ok]
        ok = far_from_plants(lat, lon)
        bg.append(pd.DataFrame({"lat": lat[ok], "lon": lon[ok]}))
    bg = pd.concat(bg).iloc[:config.GLOBAL_N_NEG_BACKGROUND].assign(neg_type="background")

    # (b) hard negatives around positives
    hard = []
    while sum(len(h) for h in hard) < config.GLOBAL_N_NEG_HARD:
        n = 8_000
        src = positives.iloc[rng.integers(0, len(positives), n)]
        dmin, dmax = config.GLOBAL_HARD_NEG_KM
        dist = np.sqrt(rng.uniform(dmin ** 2, dmax ** 2, n))      # uniform over the annulus area
        lat, lon = _destination(src["lat"].to_numpy(), src["lon"].to_numpy(), rng.uniform(0, 360, n), dist)
        ok = land_filter(lat, lon, land)
        lat, lon = lat[ok], lon[ok]
        ok = far_from_plants(lat, lon)
        hard.append(pd.DataFrame({"lat": lat[ok], "lon": lon[ok]}))
    hard = pd.concat(hard).iloc[:config.GLOBAL_N_NEG_HARD].assign(neg_type="hard")

    neg = pd.concat([bg, hard], ignore_index=True)
    neg = assign_region(neg, regions)
    neg["label"] = 0
    print("[neg] negatives per region:\n" + neg["region"].value_counts().to_string())
    return neg


def main() -> pd.DataFrame:
    rng = np.random.default_rng(config.RANDOM_SEED)
    gee.init_ee()

    regions = world_regions()
    print(f"[regions] {len(regions)} LSIB countries")
    plants = pv_plants()
    print(f"[pv] {len(plants):,} plants in inventory")

    pos_sel = select_positives(plants, regions, rng)
    interior = plant_interior_points(pos_sel["unique_id"].astype(int).tolist(), rng)
    pos = pos_sel.drop(columns=["lat", "lon"]).merge(interior, on="unique_id", how="inner")
    pos["label"] = 1
    pos["neg_type"] = ""

    neg = sample_negatives(plants, pos, regions, rng)

    pts = pd.concat([pos, neg], ignore_index=True)
    pts["pid"] = [f"G{i:06d}" for i in range(len(pts))]
    # spatial CV groups: 2 x 2 degree blocks
    pts["group_id"] = ("B" + (pts["lat"] // 2).astype(int).astype(str) + "_" + (pts["lon"] // 2).astype(int).astype(str))

    feats = gee.sample_points(pts[["pid", "lat", "lon"]], image=gee_global.training_image(),
                              bands=gee_global.TRAINING_BANDS, cache_tag="global_training")
    df = pts.merge(feats, on="pid", how="left")
    keep = ["pid", "lat", "lon", "label", "neg_type", "region", "country_na", "group_id",
            "unique_id", "area_m2", "capacity_mw", "install_date", "lc_before"] + gee_global.TRAINING_BANDS
    df = df[keep]
    df.to_csv(config.GLOBAL_DATASET_CSV, index=False)
    print(f"[done] {len(df):,} rows -> {config.GLOBAL_DATASET_CSV}")
    return df


def extract_tn_global_features() -> pd.DataFrame:
    """Global-model features for every Tamil Nadu case-study point, so the
    global model can be evaluated on exactly the same points/labels as the
    local Tamil Nadu model."""
    tn = pd.read_csv(config.DATASET_CSV)[["pid", "lat", "lon", "label", "group_id"]]
    tn["pid"] = "TN" + tn["pid"].astype(str)
    feats = gee.sample_points(tn[["pid", "lat", "lon"]], image=gee_global.training_image(),
                              bands=gee_global.TRAINING_BANDS, cache_tag="tn_global")
    df = tn.merge(feats, on="pid", how="left")
    df["pid"] = df["pid"].str[2:]
    df.to_csv(config.TN_GLOBAL_FEATURES_CSV, index=False)
    print(f"[tn-global] {len(df)} rows -> {config.TN_GLOBAL_FEATURES_CSV}")
    return df


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "tn":
        extract_tn_global_features()
    else:
        main()
