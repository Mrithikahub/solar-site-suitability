"""
Inference core shared by the FastAPI backend, notebooks and scripts.

    from model.predict import Predictor
    p = Predictor()
    p.predict(9.35, 78.38)

Feature acquisition strategy for a clicked coordinate:
  1. on-disk cache of earlier live results (rounded to 4 decimals, ~11 m)
  2. LIVE Google Earth Engine extraction of the current-epoch global stack,
     with a hard timeout (default 5 s). Restricted-mode / quota / network
     errors fall through immediately.
  3. fallback: features of the nearest cached 0.5 deg grid cell, flagged as
     source = "grid_estimate".
Every live result is cached, so the same point never costs a second request.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutTimeout
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import config
from .features import GLOBAL_FEATURE_META, IGBP_NAMES, engineer_global
from .shap_utils import explain_row, make_explainer

LIVE_CACHE_DIR = config.DATA_DIR / "cache" / "live"
LIVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
GRID_PRED_CSV = config.PROCESSED_DIR / "global_grid_predictions_0p5.csv.gz"
EARTH_R_KM = 6371.0088

# Energy model defaults (user formula): kWh/yr = area x GHI x 365 x efficiency x PR
ACRE_M2 = 4046.8564224
PANEL_EFFICIENCY = 0.20
PERFORMANCE_RATIO = 0.75
GRID_EMISSION_KG_PER_KWH = 0.475     # IEA world-average grid carbon intensity
HOUSEHOLD_KWH_PER_YEAR = 3_500       # typical annual household consumption (global mid-range)


def energy_estimate(ghi_kwh_m2_day: float, area_m2: float = ACRE_M2, efficiency: float = PANEL_EFFICIENCY,
                    performance_ratio: float = PERFORMANCE_RATIO, monthly_ghi: list[float] | None = None) -> dict:
    """Annual yield for a PV array covering `area_m2`.

        E [kWh/yr] = A [m2] x GHI [kWh/m2/day] x 365 x eta x PR
    """
    annual = area_m2 * ghi_kwh_m2_day * 365 * efficiency * performance_ratio
    out = {
        "area_m2": area_m2, "area_acres": area_m2 / ACRE_M2, "ghi_kwh_m2_day": ghi_kwh_m2_day,
        "efficiency": efficiency, "performance_ratio": performance_ratio,
        "annual_kwh": annual, "annual_mwh": annual / 1000,
        "peak_capacity_kwp": area_m2 * efficiency,          # 1 kW/m2 STC irradiance
        "specific_yield_kwh_per_kwp": annual / (area_m2 * efficiency) if area_m2 else 0.0,
        "co2_avoided_tonnes": annual * GRID_EMISSION_KG_PER_KWH / 1000,
        "households_powered": annual / HOUSEHOLD_KWH_PER_YEAR,
        "formula": "E = A × GHI × 365 × η × PR",
    }
    if monthly_ghi:
        days = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        out["monthly_kwh"] = [area_m2 * g * d * efficiency * performance_ratio for g, d in zip(monthly_ghi, days)]
    return out


def _unit_xyz(lat, lon):
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))


class Predictor:
    """Loads the global + Tamil Nadu models and answers point queries."""

    def __init__(self, live_timeout_s: float | None = None, enable_live: bool | None = None):
        bundle = joblib.load(config.MODEL_DIR / "model.pkl")
        self.pipeline = bundle["pipeline"]
        self.features = bundle["features"]
        self.model_name = bundle["model_name"]
        self.thresholds = bundle["class_thresholds"]
        self.explainer = make_explainer(self.pipeline)

        self.grid = pd.read_csv(GRID_PRED_CSV)
        self.grid_tree = cKDTree(_unit_xyz(self.grid["lat"].to_numpy(), self.grid["lon"].to_numpy()))

        self.live_timeout_s = float(live_timeout_s if live_timeout_s is not None
                                    else os.getenv("LIVE_GEE_TIMEOUT_S", "5"))
        self.enable_live = (enable_live if enable_live is not None
                            else os.getenv("LIVE_GEE", "1").lower() not in ("0", "false", "no"))
        self._pool = ThreadPoolExecutor(max_workers=4)
        self._gee_lock = threading.Lock()
        self._gee_ready = False
        self._gee_disabled_until = 0.0      # circuit breaker after restricted-mode errors
        self.last_live_error: str | None = None

        if self.enable_live:
            # warm up Earth Engine (auth + image graph) off the request path
            self._pool.submit(self._warmup)

        # headline physical suitability model (region-balanced, PU, monotonic)
        pb = joblib.load(config.MODEL_DIR / "model_physical.pkl")
        self.physical = {"pipeline": pb["pipeline"], "features": pb["features"], "name": pb["model_name"],
                         "temperature": pb.get("score_temperature", 1.0)}
        self.physical_explainer = make_explainer(self.physical["pipeline"])

        # AHP weights (Phase 2) for a per-point physical-potential score
        self.ahp_weights = None
        ahp_json = config.PROCESSED_DIR / "ahp_results.json"
        if ahp_json.exists():
            self.ahp_weights = json.loads(ahp_json.read_text())["global"]["weights"]

        # optional Tamil Nadu local model + its heatmap raster (for points inside TN)
        self.tn = None
        try:
            tnb = joblib.load(config.MODEL_DIR / "model_tn.pkl")
            stack = np.load(config.PROCESSED_DIR / "tn_grid_coords.npz")
            self.tn = {"bundle": tnb, "score": np.load(config.PROCESSED_DIR / "tn_ml_score.npy"),
                       "lats": stack["lats"], "lons": stack["lons"]}
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------ #
    # Feature acquisition
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cache_path(lat: float, lon: float) -> Path:
        return LIVE_CACHE_DIR / f"{lat:.4f}_{lon:.4f}.json"

    def _warmup(self) -> None:
        from . import gee, gee_global

        try:
            with self._gee_lock:
                if not self._gee_ready:
                    gee.init_ee()
                    self._image = gee_global.prediction_image()
                    self._gee_ready = True
        except Exception as e:  # no credentials / offline: requests will use the grid
            self.last_live_error = f"warm-up failed: {str(e).splitlines()[0][:160]}"

    def _cache_live(self, lat: float, lon: float, feats: dict, latency_s: float) -> dict:
        prov = {"source": "live_gee", "label": "Live Earth Engine extraction", "latency_s": round(latency_s, 2),
                "cached": False}
        self._cache_path(lat, lon).write_text(json.dumps({"features": feats, "provenance": prov}))
        return prov

    def _live_gee(self, lat: float, lon: float) -> dict:
        from . import gee, gee_global

        with self._gee_lock:
            if not self._gee_ready:
                gee.init_ee()
                self._image = gee_global.prediction_image()
                self._gee_ready = True
        pts = pd.DataFrame({"pid": ["live"], "lat": [lat], "lon": [lon]})
        rows = gee._sample_batch(self._image, pts, gee_global.PREDICTION_BANDS, config.GEE_SAMPLE_SCALE_M)
        return rows[0]

    def nearest_cell(self, lat: float, lon: float) -> tuple[pd.Series, float]:
        _, i = self.grid_tree.query(_unit_xyz(np.array([lat]), np.array([lon])), k=1)
        row = self.grid.iloc[int(i[0])]
        return row, _haversine_km(lat, lon, row["lat"], row["lon"])

    def get_features(self, lat: float, lon: float) -> tuple[dict, dict]:
        """Returns (raw_features, provenance)."""
        lat, lon = round(lat, 4), round(lon, 4)
        cp = self._cache_path(lat, lon)
        if cp.exists():
            d = json.loads(cp.read_text())
            return d["features"], {**d["provenance"], "cached": True}

        live_err = None
        if self.enable_live and time.time() >= self._gee_disabled_until:
            t0 = time.time()
            fut = self._pool.submit(self._live_gee, lat, lon)
            try:
                feats = fut.result(timeout=self.live_timeout_s)
                if feats.get("elevation") is None and feats.get("ghi") is None:
                    raise ValueError("no Earth Engine data at this location (ocean?)")
                return feats, self._cache_live(lat, lon, feats, time.time() - t0)
            except FutTimeout:
                live_err = f"timeout after {self.live_timeout_s:.0f}s"

                # keep the slow request alive and cache its answer when it lands,
                # so the next query for this point is served live from disk
                def _late(f, lat=lat, lon=lon, t0=t0):
                    try:
                        feats_ = f.result()
                        if feats_.get("elevation") is not None or feats_.get("ghi") is not None:
                            self._cache_live(lat, lon, feats_, time.time() - t0)
                    except Exception:
                        pass
                fut.add_done_callback(_late)
            except Exception as e:  # restricted mode, quota, auth, network
                live_err = str(e).split("\n")[0][:200]
                if any(k in live_err for k in ("Restricted", "quota", "Too Many", "429", "credentials",
                                               "not registered", "GEE_PROJECT_ID")):
                    self._gee_disabled_until = time.time() + 600   # back off for 10 min
            self.last_live_error = live_err

        row, dist_km = self.nearest_cell(lat, lon)
        feats = {k: (None if pd.isna(row[k]) else float(row[k])) for k in
                 ["elevation", "slope", "aspect", "ghi", "temperature", "cloud_cover", "accessibility",
                  "ndvi", "ndbi", "lst", "landcover", "nightlights", "population", "biome", "worldcover"]}
        prov = {"source": "grid_estimate", "label": "Grid estimate (nearest 0.5° cell)",
                "cell_lat": float(row["lat"]), "cell_lon": float(row["lon"]),
                "cell_distance_km": round(dist_km, 1), "country": row.get("country_na"),
                "region": row.get("region"), "live_error": live_err, "cached": False,
                "live_enabled": self.enable_live}
        return feats, prov

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def classify(self, score: float) -> str:
        if score >= self.thresholds["High"]:
            return "High"
        if score >= self.thresholds["Medium"]:
            return "Medium"
        return "Low"

    def tn_local_score(self, lat: float, lon: float) -> float | None:
        if self.tn is None:
            return None
        lats, lons = self.tn["lats"], self.tn["lons"]
        if not (lats[-1] <= lat <= lats[0] and lons[0] <= lon <= lons[-1]):
            return None
        r = int(np.clip(np.searchsorted(-lats, -lat), 0, len(lats) - 1))
        c = int(np.clip(np.searchsorted(lons, lon), 0, len(lons) - 1))
        v = self.tn["score"][r, c]
        return None if np.isnan(v) else float(v)

    def predict(self, lat: float, lon: float, area_m2: float = ACRE_M2, efficiency: float = PANEL_EFFICIENCY,
                performance_ratio: float = PERFORMANCE_RATIO, top_shap: int | None = None) -> dict:
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("latitude must be in [-90, 90] and longitude in [-180, 180]")
        feats, prov = self.get_features(lat, lon)
        if prov["source"] == "grid_estimate" and prov["cell_distance_km"] > 60:
            raise LookupError("This location is not on land covered by the model (open water or outside "
                              f"{config.GLOBAL_LAT_RANGE[0]:.0f}°..{config.GLOBAL_LAT_RANGE[1]:.0f}° latitude).")

        from .physical import exclusion_message, exclusion_reason, temper

        raw = pd.DataFrame([{**feats, "lat": lat, "lon": lon}])
        X = engineer_global(raw, epoch="current")

        # 1) headline: physical suitability (tempered probability, exclusions -> 0)
        Xp = X[self.physical["features"]]
        T = self.physical["temperature"]
        p_phys = float(self.physical["pipeline"].predict_proba(Xp)[0, 1])
        model_score = float(temper(p_phys, T))
        excluded = exclusion_reason(feats.get("landcover"), feats.get("slope"))
        score = 0.0 if excluded else model_score
        expl = explain_row(self.physical_explainer, self.physical["pipeline"], Xp, GLOBAL_FEATURE_META,
                           top=top_shap, transform=lambda q: float(temper(q, T)) / 100.0)

        # 2) development likelihood: the siting model (physical + access / lights / population)
        Xd = X[self.features]
        dev = float(self.pipeline.predict_proba(Xd)[0, 1] * 100)
        dev_expl = explain_row(self.explainer, self.pipeline, Xd, GLOBAL_FEATURE_META, top=6)

        ghi = feats.get("ghi")
        energy = energy_estimate(ghi, area_m2, efficiency, performance_ratio) if ghi is not None else None
        lc = feats.get("landcover")
        wc = feats.get("worldcover")
        display = {
            "ghi": ghi, "temperature": feats.get("temperature"), "cloud_cover": feats.get("cloud_cover"),
            "elevation": feats.get("elevation"), "slope": feats.get("slope"), "aspect": feats.get("aspect"),
            "ndvi": feats.get("ndvi"), "ndbi": feats.get("ndbi"), "lst": feats.get("lst"),
            "accessibility": feats.get("accessibility"), "nightlights": feats.get("nightlights"),
            "population": feats.get("population"),
            "landcover": IGBP_NAMES.get(int(round(lc))) if lc is not None else None,
            "worldcover": config.WORLDCOVER_CLASSES.get(int(round(wc))) if wc is not None else None,
        }
        out = {
            "lat": lat, "lon": lon,
            "score": round(score, 1), "class": self.classify(score),
            "excluded": excluded, "exclusion_message": exclusion_message(excluded),
            "model_score": round(model_score, 1),
            "model": {"name": self.physical["name"], "variant": "physical suitability (region-balanced PU)",
                      "thresholds": self.thresholds},
            "features": display,
            "feature_meta": {k: GLOBAL_FEATURE_META.get(k, {}) for k in display},
            "shap": expl,
            "development_shap": dev_expl,
            "energy": energy,
            "provenance": prov,
        }
        scores = {
            "suitability": {
                "score": round(score, 1), "class": self.classify(score), "label": "Suitability",
                "description": "How physically suited the land is to utility-scale PV - sunshine, cloud, slope, "
                               "vegetation and land cover only." + (f" Excluded: {excluded}." if excluded else "")},
            "development": {
                "score": round(dev, 1), "class": self.classify(dev), "label": "Development likelihood",
                "description": "How closely the site resembles places where plants have historically been built "
                               "- near roads, grids and demand. Not a measure of physical suitability."},
        }
        if self.ahp_weights is not None:
            from .ahp import GLOBAL_CRITERIA, global_scores, overlay

            d = {k: np.array([np.nan if feats.get(k) is None else feats[k]], float)
                 for k in ["ghi", "slope", "landcover", "accessibility", "ndvi", "temperature"]}
            r = global_scores(d)
            a = float(overlay(r["scores"], r["constraint"], GLOBAL_CRITERIA,
                              np.array([self.ahp_weights[c] for c in GLOBAL_CRITERIA]))[0])
            scores["ahp"] = {"score": round(a, 1), "class": self.classify(a), "label": "AHP score",
                             "description": "Classic expert-weighted GIS overlay of irradiance, slope, land cover, "
                                            "access, vegetation and temperature, with exclusions."}
        out["scores"] = scores

        tn = self.tn_local_score(lat, lon)
        if tn is not None:
            out["tamil_nadu_local"] = {"score": round(tn, 1), "class": self.classify(tn),
                                       "model": self.tn["bundle"]["model_name"],
                                       "label": "High-res regional model",
                                       "note": "Available for select regions: 10-30 m imagery and local grid distances (Tamil Nadu, India)"}
        return out
