"""
OpenStreetMap data via the Overpass API.

* Solar plant footprints  -> positive training labels
* Major roads             -> distance-to-road feature (site access / construction)
* Power lines, substations-> distance-to-grid features (evacuation of power)

Distances are computed in a metric CRS (UTM 44N for Tamil Nadu). Lines are
densified to vertices every ~100 m and indexed with a KD-tree, which gives
nearest-line distances accurate to +/- 50 m while answering thousands of
queries in milliseconds - fast enough for the live prediction API.
"""
from __future__ import annotations

import json
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiPolygon, Polygon, shape
from shapely.ops import linemerge, polygonize, unary_union

from . import config

_TO_METRIC = Transformer.from_crs(4326, config.METRIC_EPSG, always_xy=True)


# --------------------------------------------------------------------------- #
# Overpass helper
# --------------------------------------------------------------------------- #
def overpass(query: str, cache_name: str | None = None, timeout: int = 600) -> list[dict]:
    """Run an Overpass QL query against the first responsive mirror (cached)."""
    cache = config.CACHE_DIR / f"overpass_{cache_name}.json" if cache_name else None
    if cache and cache.exists():
        return json.loads(cache.read_text())
    last = None
    for attempt in range(2):
        for url in config.OVERPASS_URLS:
            try:
                r = requests.post(url, data={"data": query}, timeout=timeout,
                                  headers={"User-Agent": config.HTTP_USER_AGENT})
                r.raise_for_status()
                els = r.json()["elements"]
                if cache:
                    cache.write_text(json.dumps(els))
                return els
            except Exception as e:
                last = e
                time.sleep(5)
    raise RuntimeError(f"All Overpass mirrors failed: {last}")


def _bbox_str(bounds, pad: float = 0.0) -> str:
    """Overpass bbox order is (south, west, north, east)."""
    min_lon, min_lat, max_lon, max_lat = bounds
    return f"{min_lat - pad},{min_lon - pad},{max_lat + pad},{max_lon + pad}"


# --------------------------------------------------------------------------- #
# Solar plant polygons (labels)
# --------------------------------------------------------------------------- #
def _relation_to_polygon(rel: dict):
    """Assemble an OSM multipolygon relation from its member way geometries."""
    outers, inners = [], []
    for m in rel.get("members", []):
        if m.get("type") != "way" or "geometry" not in m:
            continue
        line = LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
        (inners if m.get("role") == "inner" else outers).append(line)
    if not outers:
        return None
    outer_polys = list(polygonize(linemerge(outers)))
    inner_polys = list(polygonize(linemerge(inners))) if inners else []
    if not outer_polys:
        return None
    geom = unary_union(outer_polys)
    if inner_polys:
        geom = geom.difference(unary_union(inner_polys))
    return geom if not geom.is_empty else None


def _way_to_polygon(way: dict):
    coords = [(p["lon"], p["lat"]) for p in way.get("geometry", [])]
    if len(coords) < 4 or coords[0] != coords[-1]:
        return None
    poly = Polygon(coords)
    return poly if poly.is_valid and poly.area > 0 else poly.buffer(0)


def fetch_solar_sites(region_polygon) -> gpd.GeoDataFrame:
    """Solar plant footprints inside the region, merged into individual *sites*.

    Plant boundaries (power=plant + plant:source=solar) and ground-mounted
    panel-array polygons (power=generator + generator:source=solar) are merged;
    touching / nearby polygons (< 50 m apart) are dissolved into one site so a
    single park is never split into several "independent" samples.
    """
    minx, miny, maxx, maxy = region_polygon.bounds
    q = f"""
    [out:json][timeout:300];
    (
      way["power"="plant"]["plant:source"="solar"]({_bbox_str((minx, miny, maxx, maxy))});
      relation["power"="plant"]["plant:source"="solar"]({_bbox_str((minx, miny, maxx, maxy))});
      way["power"="generator"]["generator:source"="solar"]({_bbox_str((minx, miny, maxx, maxy))});
      relation["power"="generator"]["generator:source"="solar"]({_bbox_str((minx, miny, maxx, maxy))});
    );
    out geom tags;
    """
    els = overpass(q, cache_name="solar_geom")
    recs = []
    for e in els:
        geom = _way_to_polygon(e) if e["type"] == "way" else _relation_to_polygon(e)
        if geom is None or geom.is_empty:
            continue
        tags = e.get("tags", {})
        recs.append({
            "osm_id": f"{e['type']}/{e['id']}",
            "kind": tags.get("power"),
            "name": tags.get("name", ""),
            "operator": tags.get("operator", ""),
            "start_date": tags.get("start_date", ""),
            "geometry": geom,
        })
    raw = gpd.GeoDataFrame(recs, crs=4326)
    raw = raw[raw.intersects(region_polygon)].to_crs(config.METRIC_EPSG)

    # dissolve into sites: 25 m outward buffer joins neighbours, then undo it
    merged = unary_union(raw.geometry.buffer(25)).buffer(-25)
    parts = list(merged.geoms) if isinstance(merged, MultiPolygon) else [merged]
    sites = gpd.GeoDataFrame({"geometry": parts}, crs=config.METRIC_EPSG)
    sites["area_m2"] = sites.area
    sites = sites[sites["area_m2"] >= config.MIN_SOLAR_AREA_M2].reset_index(drop=True)

    # carry over the most informative name / start date from constituent polygons
    joined = gpd.sjoin(raw, sites[["geometry"]], how="inner", predicate="intersects")
    meta = []
    for idx in sites.index:
        sub = joined[joined["index_right"] == idx]
        plants = sub[sub["kind"] == "plant"]
        src = plants if len(plants) else sub
        names = [n for n in src["name"] if n]
        dates = [d for d in src["start_date"] if d]
        meta.append({"name": names[0] if names else "", "start_date": dates[0] if dates else "",
                     "n_osm_polygons": len(sub)})
    sites = pd.concat([sites, pd.DataFrame(meta)], axis=1)
    sites["site_id"] = [f"S{i:04d}" for i in range(len(sites))]
    return sites.to_crs(4326)


# --------------------------------------------------------------------------- #
# Infrastructure (roads, power lines, substations)
# --------------------------------------------------------------------------- #
def _densify_ways(ways: list[dict], step_m: float = 100.0) -> np.ndarray:
    """Convert Overpass way geometries into metric vertices spaced <= step_m."""
    chunks = []
    for w in ways:
        g = w.get("geometry")
        if not g or len(g) < 2:
            continue
        lon = np.array([p["lon"] for p in g])
        lat = np.array([p["lat"] for p in g])
        x, y = _TO_METRIC.transform(lon, lat)
        xy = np.column_stack([x, y])
        seg = np.diff(xy, axis=0)
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        pts = [xy[:1]]
        for i, L in enumerate(seg_len):
            n = max(1, int(np.ceil(L / step_m)))
            t = np.arange(1, n + 1)[:, None] / n
            pts.append(xy[i] + t * seg[i])
        chunks.append(np.vstack(pts))
    return np.vstack(chunks).astype(np.float32) if chunks else np.empty((0, 2), np.float32)


def _tiled_query(template: str, bounds, name: str, nx: int = 2, ny: int = 2, pad: float = 0.3) -> list[dict]:
    """Split a large bbox into tiles so each Overpass request stays small."""
    min_lon, min_lat, max_lon, max_lat = bounds
    min_lon, min_lat, max_lon, max_lat = min_lon - pad, min_lat - pad, max_lon + pad, max_lat + pad
    xs = np.linspace(min_lon, max_lon, nx + 1)
    ys = np.linspace(min_lat, max_lat, ny + 1)
    out, seen = [], set()
    for i in range(nx):
        for j in range(ny):
            bb = f"{ys[j]},{xs[i]},{ys[j + 1]},{xs[i + 1]}"
            els = overpass(template.format(bbox=bb), cache_name=f"{name}_{i}{j}")
            for e in els:
                if e["id"] not in seen:
                    seen.add(e["id"])
                    out.append(e)
    return out


def build_infrastructure_index(bounds, verbose: bool = True) -> None:
    """Download roads / power lines / substations around `bounds` and save
    densified vertex arrays used for distance lookups."""
    roads_q = """
    [out:json][timeout:600];
    way["highway"~"^(motorway|trunk|primary|secondary|tertiary)(_link)?$"]({bbox});
    out skel geom qt;
    """
    lines_q = """
    [out:json][timeout:600];
    way["power"~"^(line|minor_line)$"]({bbox});
    out skel geom qt;
    """
    subs_q = """
    [out:json][timeout:300];
    nwr["power"="substation"]({bbox});
    out center qt;
    """
    roads = _tiled_query(roads_q, bounds, "roads", 3, 3)
    road_xy = _densify_ways(roads)
    np.save(config.ROADS_KDTREE, road_xy)
    if verbose:
        print(f"[osm] roads: {len(roads):,} ways -> {len(road_xy):,} vertices")

    lines = _tiled_query(lines_q, bounds, "powerlines", 2, 2)
    line_xy = _densify_ways(lines)
    np.save(config.POWERLINES_KDTREE, line_xy)
    if verbose:
        print(f"[osm] power lines: {len(lines):,} ways -> {len(line_xy):,} vertices")

    subs = _tiled_query(subs_q, bounds, "substations", 2, 2)
    lon = np.array([e.get("lon", e.get("center", {}).get("lon")) for e in subs], dtype=float)
    lat = np.array([e.get("lat", e.get("center", {}).get("lat")) for e in subs], dtype=float)
    ok = ~(np.isnan(lon) | np.isnan(lat))
    x, y = _TO_METRIC.transform(lon[ok], lat[ok])
    np.save(config.SUBSTATIONS_NPY, np.column_stack([x, y]).astype(np.float32))
    if verbose:
        print(f"[osm] substations: {ok.sum():,}")


class DistanceIndex:
    """Nearest-distance lookups (km) to roads, power lines and substations."""

    def __init__(self):
        self.trees = {
            "dist_road_km": cKDTree(np.load(config.ROADS_KDTREE)),
            "dist_powerline_km": cKDTree(np.load(config.POWERLINES_KDTREE)),
            "dist_substation_km": cKDTree(np.load(config.SUBSTATIONS_NPY)),
        }

    def __call__(self, lat, lon) -> pd.DataFrame:
        x, y = _TO_METRIC.transform(np.atleast_1d(lon), np.atleast_1d(lat))
        xy = np.column_stack([x, y])
        return pd.DataFrame({k: t.query(xy, k=1)[0] / 1000.0 for k, t in self.trees.items()})


def region_polygon_from_geojson(gj: dict):
    """Region as a (Multi)Polygon. Earth Engine can return the simplified
    boundary as a GeometryCollection that also holds stray line fragments,
    so only polygonal parts are kept."""
    parts = []
    for f in gj["features"]:
        g = shape(f["geometry"])
        geoms = getattr(g, "geoms", [g])
        parts.extend(p for p in geoms if p.geom_type in ("Polygon", "MultiPolygon"))
    return unary_union(parts)
