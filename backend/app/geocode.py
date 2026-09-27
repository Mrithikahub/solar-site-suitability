"""
Place search for the map: Photon (primary) + Nominatim (fallback), proxied
through the backend so results are cached and each service's usage policy is
respected (identifying User-Agent, Photon <= ~4 req/s, Nominatim <= 1 req/s).

Pipeline for a query
  1. raw coordinates ("51.5294, -0.1727", "51.53 N 0.17 W") -> returned directly
  2. Photon search (OSM-based, typo-tolerant, good at POIs)
  3. if no candidate covers every query word, extra query variants:
       - without generic category words  ("lords stadium london" -> "lords london")
       - with the possessive restored    ("lords"                -> "lord's")
  4. Nominatim for the original + possessive query if coverage is still poor
  5. all candidates are re-ranked by token match on name / address, a bonus
     when a category word matches the OSM tag (e.g. "stadium" ~ leisure=stadium),
     and a penalty for minor features (bus stops, shops, artworks)

Each result carries kind = "point" (POI / building / address -> fly in and
analyse) or "area" (city / region / country with a bounding box -> fit bounds).
"""
from __future__ import annotations

import difflib
import math
import os
import re
import threading
import time
import unicodedata

import httpx

from .cache import TTLCache

PHOTON_URL = os.getenv("PHOTON_URL", "https://photon.komoot.io/api/")
NOMINATIM_URL = os.getenv("NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")
USER_AGENT = os.getenv("GEOCODER_USER_AGENT", "solarsite/1.0 (+https://github.com/Mrithikahub/solar-site-suitability)")

_cache = TTLCache(maxsize=8192, ttl_s=7 * 24 * 3600)
_rate = {"photon": (threading.Lock(), 0.25), "nominatim": (threading.Lock(), 1.05)}
_last = {"photon": 0.0, "nominatim": 0.0}

AREA_TYPES = {"country", "state", "region", "province", "county", "district", "city", "town", "village",
              "municipality", "borough", "suburb", "island", "archipelago", "continent", "administrative",
              "hamlet", "quarter", "neighbourhood"}
MINOR_TYPES = {"bus_stop", "platform", "stop_position", "tram_stop", "artwork", "vending_machine", "bench",
               "shop", "ticket", "houseware", "office", "atm", "post_box", "waste_basket", "information"}
# query word -> OSM values it should match
CATEGORY_WORDS = {
    "stadium": {"stadium", "sports_centre", "pitch"}, "ground": {"stadium", "pitch", "recreation_ground"},
    "arena": {"stadium", "sports_centre"}, "park": {"park", "national_park", "nature_reserve", "plant", "locality"},
    "solar": {"plant", "generator", "locality"}, "farm": {"plant", "farm", "farmland"},
    "tower": {"tower", "attraction"}, "airport": {"aerodrome", "airport"}, "station": {"station", "train_station"},
    "museum": {"museum"}, "university": {"university", "college"}, "hospital": {"hospital"},
    "temple": {"place_of_worship", "temple"}, "church": {"place_of_worship", "church", "cathedral"},
    "mosque": {"place_of_worship", "mosque"}, "palace": {"palace", "castle", "attraction"},
    "fort": {"fort", "castle"}, "lake": {"water", "lake", "reservoir"}, "beach": {"beach"}, "bridge": {"bridge"},
    "mountain": {"peak", "mountain_range"}, "mount": {"peak"}, "desert": {"desert"}, "dam": {"dam"},
    "zoo": {"zoo"}, "mall": {"mall"}, "port": {"harbour", "port"}, "island": {"island"},
}
STOPWORDS = {"the", "of", "in", "at", "near", "and", "de", "la", "le"}
STREET_WORDS = {"street", "st", "road", "rd", "avenue", "ave", "lane", "way", "drive", "boulevard", "highway"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = s.lower().replace("'", "").replace("’", "")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _tokens(s: str) -> list[str]:
    return [t for t in _norm(s).split() if t and t not in STOPWORDS]


def _throttle(service: str) -> None:
    lock, interval = _rate[service]
    with lock:
        wait = interval - (time.time() - _last[service])
        if wait > 0:
            time.sleep(wait)
        _last[service] = time.time()


def _pretty(v: str | None) -> str:
    return (v or "place").replace("_", " ")


_COORD = re.compile(
    r"^\s*(-?\d{1,3}(?:\.\d+)?)\s*°?\s*([NSns])?\s*[,;\s]\s*(-?\d{1,3}(?:\.\d+)?)\s*°?\s*([EWew])?\s*$")


def parse_coordinates(q: str) -> dict | None:
    m = _COORD.match(q)
    if not m:
        return None
    lat, ns, lon, ew = float(m.group(1)), m.group(2), float(m.group(3)), m.group(4)
    if ns and ns.upper() == "S":
        lat = -abs(lat)
    if ew and ew.upper() == "W":
        lon = -abs(lon)
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return {"name": f"{lat:.5f}, {lon:.5f}", "display_name": "Coordinates", "lat": lat, "lon": lon,
            "type": "coordinates", "category": "coordinates", "kind": "point", "bbox": None, "zoom": 16,
            "source": "coordinates"}


AREA_ZOOM = {"continent": 3, "country": 5, "state": 6, "region": 6, "province": 6, "county": 9, "district": 10,
             "island": 10, "archipelago": 7, "city": 11, "municipality": 11, "administrative": 10, "town": 12,
             "borough": 13, "suburb": 14, "village": 14, "hamlet": 15, "quarter": 15, "neighbourhood": 15}


def _zoom_for(kind: str, bbox: list[float] | None, place_type: str = "") -> int:
    """Suggested zoom: areas by their type (used when there is no bounding box),
    points by footprint size (a 10 km solar park gets a wider view than a tower)."""
    if kind == "area":
        return AREA_ZOOM.get(place_type, 10)
    if not bbox:
        return 17
    s, n, w, e = bbox
    span_km = max(abs(n - s), abs(e - w) * math.cos(math.radians((n + s) / 2))) * 111
    if span_km > 8:
        return 13
    if span_km > 3:
        return 14
    if span_km > 1:
        return 15
    return 17


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
def _photon(q: str, limit: int = 10) -> list[dict]:
    key = ("photon", q.lower(), limit)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    _throttle("photon")
    r = httpx.get(PHOTON_URL, params={"q": q, "limit": limit, "lang": "en"},
                  headers={"User-Agent": USER_AGENT}, timeout=10)
    r.raise_for_status()
    out = []
    for f in r.json().get("features", []):
        p = f.get("properties", {})
        lon, lat = f["geometry"]["coordinates"][:2]
        ext = p.get("extent")                       # [minlon, maxlat, maxlon, minlat]
        bbox = [ext[3], ext[1], ext[0], ext[2]] if ext else None   # -> [S, N, W, E]
        ptype = p.get("type") or ""
        value = p.get("osm_value") or ""
        is_area = (ptype in AREA_TYPES or value in AREA_TYPES) and p.get("osm_key") in ("place", "boundary")
        street = " ".join(x for x in [p.get("housenumber"), p.get("street")] if x)
        addr = ", ".join(dict.fromkeys(x for x in [street, p.get("district"), p.get("city"), p.get("county"),
                                                    p.get("state"), p.get("country")] if x and x != p.get("name")))
        name = p.get("name") or street or p.get("city") or p.get("country") or "Unnamed place"
        kind = "area" if is_area else "point"
        out.append({"name": name, "display_name": addr or name, "lat": lat, "lon": lon, "type": _pretty(value or ptype),
                    "category": p.get("osm_key") or "", "osm_value": value, "kind": kind, "bbox": bbox,
                    "zoom": _zoom_for(kind, bbox, value if value in AREA_ZOOM else ptype), "source": "photon",
                    "_context": " ".join(x for x in [p.get("city"), p.get("district"), p.get("county"), p.get("state"),
                                                     p.get("country"), p.get("street")] if x)})
    _cache.set(key, out)
    return out


def _nominatim(q: str, limit: int = 6) -> list[dict]:
    key = ("nominatim", q.lower(), limit)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    _throttle("nominatim")
    r = httpx.get(NOMINATIM_URL, params={"q": q, "format": "jsonv2", "limit": limit, "addressdetails": 0},
                  headers={"User-Agent": USER_AGENT, "Accept-Language": "en"}, timeout=10)
    r.raise_for_status()
    out = []
    for it in r.json():
        bb = it.get("boundingbox")
        bbox = [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])] if bb else None   # S, N, W, E
        atype = it.get("addresstype") or it.get("type") or ""
        is_area = it.get("category") in ("place", "boundary") and atype in AREA_TYPES
        kind = "area" if is_area else "point"
        disp = it.get("display_name", "")
        out.append({"name": it.get("name") or disp.split(",")[0], "display_name": ", ".join(disp.split(", ")[1:]) or disp,
                    "lat": float(it["lat"]), "lon": float(it["lon"]), "type": _pretty(it.get("type")),
                    "category": it.get("category") or "", "osm_value": it.get("type") or "", "kind": kind,
                    "bbox": bbox, "zoom": _zoom_for(kind, bbox, atype), "source": "nominatim", "_context": disp})
    _cache.set(key, out)
    return out


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def _fuzzy_in(tok: str, pool: list[str]) -> bool:
    return any(tok == t or (len(tok) > 3 and difflib.SequenceMatcher(None, tok, t).ratio() >= 0.84) for t in pool)


def _score(c: dict, q_tokens: list[str], rank: int, n: int) -> tuple[float, bool]:
    name_t = _tokens(c["name"])
    ctx_t = _tokens(c.get("_context", "") + " " + c.get("display_name", ""))
    value = c.get("osm_value", "")
    total, covered = 0.0, 0
    for t in q_tokens:
        if t in name_t:
            total += 2.0
            covered += 1
        elif _fuzzy_in(t, name_t):
            total += 1.6
            covered += 1
        elif t in CATEGORY_WORDS and value in CATEGORY_WORDS[t]:
            total += 1.5
            covered += 1
        elif t in ctx_t or _fuzzy_in(t, ctx_t):
            total += 1.0
            covered += 1
    s = total / max(1, len(q_tokens))
    if value in MINOR_TYPES or c.get("category") in ("shop", "office"):
        s -= 0.45
    if c.get("category") == "highway" and not (set(q_tokens) & STREET_WORDS):
        s -= 0.5                                  # a street named after the place is rarely what was meant
    # names made of words the user never typed are less likely (e.g. "House of Lords Library")
    extra = [t for t in name_t if t not in q_tokens and not _fuzzy_in(t, q_tokens) and t not in CATEGORY_WORDS]
    s -= 0.08 * len(extra)
    s += 0.25 * (1 - rank / max(1, n))              # keep the provider's own relevance as a prior
    return s, covered == len(q_tokens)


def _rank(cands: list[dict], q: str) -> list[dict]:
    qt = _tokens(q)
    scored, seen = [], set()
    for i, c in enumerate(cands):
        key = (_norm(c["name"]), round(c["lat"], 3), round(c["lon"], 3))
        if key in seen:
            continue
        seen.add(key)
        s, full = _score(c, qt, i, len(cands))
        scored.append((s + (0.3 if full else 0.0), full, c))
    scored.sort(key=lambda x: -x[0])
    return [c for _, _, c in scored]


def _correct_categories(q: str) -> str:
    """Fix misspelled category words ("stadum" -> "stadium") so variants and
    category bonuses still apply."""
    fixed = []
    for t in q.split():
        n = _norm(t)
        if n and n not in CATEGORY_WORDS and len(n) >= 4:
            m = difflib.get_close_matches(n, CATEGORY_WORDS.keys(), n=1, cutoff=0.8)
            fixed.append(m[0] if m else t)
        else:
            fixed.append(t)
    return " ".join(fixed)


def _variants(q: str) -> list[str]:
    toks = q.split()
    out = []
    no_cat = [t for t in toks if _norm(t) not in CATEGORY_WORDS]
    if len(no_cat) != len(toks) and no_cat:
        out.append(" ".join(no_cat))
    poss = [re.sub(r"^([A-Za-z]{3,})s$", r"\1's", t) if _norm(t) not in CATEGORY_WORDS else t for t in toks]
    if poss != toks:
        out.append(" ".join(poss))
    return out


def _covers(cands: list[dict], q: str) -> bool:
    """True if one of the best-ranked candidates matches every query word."""
    qt = _tokens(q)
    top = _rank(cands, q)[:3]
    return any(_score(c, qt, 0, 1)[1] for c in top)


# --------------------------------------------------------------------------- #
# Curated famous solar parks (ranked first when the query matches)
# --------------------------------------------------------------------------- #
# Coordinates: centroid of the largest panel polygon in the Kruitwagen et al.
# (2021) inventory where available, otherwise the OSM site location (parks
# completed after the 2018 inventory). Capacities are published nameplate figures.
# `featured` parks appear as quick-pick chips on the map (a globally balanced set);
# every park is searchable.
FAMOUS_PARKS = [
    {"name": "Bhadla Solar Park", "aliases": ["bhadla", "badla"], "lat": 27.4745, "lon": 71.9726,
     "place": "Jodhpur, Rajasthan, India", "capacity_mw": 2245, "featured": True},
    {"name": "Kamuthi Solar Power Project", "aliases": ["kamuthi"], "lat": 9.3322, "lon": 78.3899,
     "place": "Ramanathapuram, Tamil Nadu, India", "capacity_mw": 648},
    {"name": "Pavagada Solar Park", "aliases": ["pavagada", "shakti sthala"], "lat": 14.2382, "lon": 77.4606,
     "place": "Tumkur, Karnataka, India", "capacity_mw": 2050},
    {"name": "Benban Solar Park", "aliases": ["benban"], "lat": 24.4110, "lon": 32.7102,
     "place": "Aswan, Egypt", "capacity_mw": 1650, "featured": True},
    {"name": "Tengger Desert Solar Park", "aliases": ["tengger"], "lat": 37.5592, "lon": 105.0365,
     "place": "Zhongwei, Ningxia, China", "capacity_mw": 1547, "featured": True},
    {"name": "Noor Ouarzazate Solar Complex", "aliases": ["noor", "ouarzazate"], "lat": 31.0400, "lon": -6.8635,
     "place": "Ouarzazate, Morocco", "capacity_mw": 580, "featured": True},
    {"name": "Solar Star", "aliases": ["solar star"], "lat": 34.8276, "lon": -118.4408,
     "place": "Rosamond, California, USA", "capacity_mw": 579, "featured": True},
    {"name": "Kurnool Ultra Mega Solar Park", "aliases": ["kurnool"], "lat": 15.6724, "lon": 78.2923,
     "place": "Kurnool, Andhra Pradesh, India", "capacity_mw": 1000},
    {"name": "Rewa Ultra Mega Solar", "aliases": ["rewa"], "lat": 24.4774, "lon": 81.5667,
     "place": "Rewa, Madhya Pradesh, India", "capacity_mw": 750},
    {"name": "Mohammed bin Rashid Al Maktoum Solar Park", "aliases": ["mohammed bin rashid", "maktoum", "mbr solar"],
     "lat": 24.7590, "lon": 55.3829, "place": "Dubai, United Arab Emirates", "capacity_mw": None, "featured": True},
    {"name": "Villanueva Solar Park", "aliases": ["villanueva"], "lat": 25.5992, "lon": -103.0323,
     "place": "Viesca, Coahuila, Mexico", "capacity_mw": 828, "featured": True},
    {"name": "Cestas Solar Farm", "aliases": ["cestas"], "lat": 44.7255, "lon": -0.8156,
     "place": "Cestas, Gironde, France", "capacity_mw": 300, "featured": True},
    {"name": "Topaz Solar Farm", "aliases": ["topaz"], "lat": 35.3687, "lon": -120.0326,
     "place": "San Luis Obispo County, California, USA", "capacity_mw": 550},
    {"name": "Longyangxia Solar Park", "aliases": ["longyangxia"], "lat": 36.0238, "lon": 100.5128,
     "place": "Gonghe, Qinghai, China", "capacity_mw": 850},
]
_PARK_GENERIC = {"solar", "park", "plant", "power", "project", "ultra", "mega", "farm", "complex", "pv"}


def famous_parks(featured_only: bool = True) -> list[dict]:
    return [_park_result(p) for p in FAMOUS_PARKS if p.get("featured") or not featured_only]


def _park_result(p: dict) -> dict:
    size = f"{p['capacity_mw']:,} MW solar park" if p.get("capacity_mw") else "Multi-phase solar park"
    return {"name": p["name"], "display_name": f"{size}, {p['place']}",
            "lat": p["lat"], "lon": p["lon"], "type": "solar park", "category": "power", "osm_value": "plant",
            "kind": "point", "bbox": None, "zoom": 15, "source": "curated", "capacity_mw": p["capacity_mw"]}


def _match_parks(q: str) -> list[dict]:
    """Parks whose alias appears in the query (typo-tolerant), or whose alias
    starts with what has been typed so far (autocomplete, >= 3 characters)."""
    qt = [t for t in _tokens(q) if t not in _PARK_GENERIC]
    qn = _norm(q)
    out = []
    for p in FAMOUS_PARKS:
        hit = False
        for alias in p["aliases"]:
            at = alias.split()
            if all(_fuzzy_in(a, qt) for a in at if a not in _PARK_GENERIC) and any(a not in _PARK_GENERIC for a in at):
                hit = True
            elif " " in alias and alias in qn:
                hit = True
            elif len(qn) >= 3 and len(qt) == 1 and alias.startswith(qt[0]):
                hit = True
        if hit:
            out.append(_park_result(p))
    return out


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def search(q: str, limit: int = 5) -> list[dict]:
    q = " ".join(q.split())[:200]
    key = ("search", q.lower(), limit)
    hit = _cache.get(key)
    if hit is not None:
        return hit

    coord = parse_coordinates(q)
    if coord:
        return [coord]

    q_fixed = _correct_categories(q)
    cands: list[dict] = []
    try:
        cands = _photon(q_fixed)
    except Exception:
        cands = []
    if not _covers(cands, q_fixed):
        for v in _variants(q_fixed):
            try:
                cands += _photon(v)
            except Exception:
                pass
    if not cands or not _covers(cands, q_fixed):
        for v in [q_fixed] + [x for x in _variants(q_fixed) if "'" in x]:
            try:
                cands += _nominatim(v)
            except Exception:
                pass

    parks = _match_parks(q)
    ranked = _rank(cands, q_fixed)
    # drop geocoder duplicates of a curated park (e.g. OSM "Badla Solar Park")
    ranked = [c for c in ranked
              if not any(abs(c["lat"] - p["lat"]) < 0.15 and abs(c["lon"] - p["lon"]) < 0.15
                         and c.get("osm_value") in ("plant", "locality", "industrial", "generator") for p in parks)]
    out = parks + [{k: v for k, v in c.items() if not k.startswith("_")} for c in ranked]
    out = out[:limit]
    _cache.set(key, out)
    return out
