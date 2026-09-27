"""
Physical suitability model - the headline "Suitability" score.

It answers "is this land physically good for utility-scale PV?" rather than
"does this look like where developers have built?". Plain supervised learning
on plant / no-plant labels cannot answer that, because the labels are shaped
by economics: plants cluster near grids and demand (cloudy Europe, Japan,
eastern China), while the world's sunniest deserts contain almost none. The
model therefore combines four ideas:

  1. Physical predictors only - climate, terrain and land surface; no night
     lights, travel time or population.
  2. Region-balanced sample weights - every world region contributes the same
     total weight, split equally between plants and non-plants (capped so a
     region with 16 plants cannot dominate).
  3. Positive-unlabelled (PU) learning with reliable negatives - a point
     without a plant is *unlabelled*, not *unsuitable*. Following the classic
     two-step PU approach, negatives whose physical plausibility is high
     (the AHP overlay without the accessibility criterion, evaluated on the
     pre-construction epoch, >= PU_THRESHOLD) are removed from this model's
     training set; only reliable negatives (forest, ice, wetland, steep,
     urban or low-irradiance land) are kept.
  4. Monotonic constraints in XGBoost for physically known effects: more
     irradiance / less cloud is never worse; steeper slope, denser vegetation,
     forest, water, wetland, snow and urban land are never better.
  5. A shallow, regularised booster (depth 3) plus logit temperature scaling,
     because PU probabilities are not calibrated and otherwise saturate.

At prediction time the standard MCDA exclusion mask is applied (water,
wetland, snow/ice, urban land, slope > 15 deg), exactly as in the AHP.

    python -m model.physical
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config
from .features import engineer_global

PHYSICAL_MODEL_PKL = config.MODEL_DIR / "model_physical.pkl"
PHYSICAL_JSON = config.PROCESSED_DIR / "physical_model.json"
SANITY_FEATURES_CSV = config.PROCESSED_DIR / "sanity_site_features.csv"
HUMAN_COLS = ["log_access", "log_lights", "log_pop"]
MAX_WEIGHT_RATIO = 10.0          # cap so 16 plants in one region can't dominate
PU_THRESHOLD = 60.0              # negatives with physical plausibility >= this are "unlabelled"
SCORE_TEMPERATURE = 2.0          # logit temperature: PU probabilities are uncalibrated and saturate
                                 # near 0/1; score = sigmoid(logit(p) / T) keeps the ranking (AUC)
                                 # but restores gradation for a 0-100 headline score

MONOTONE = {
    "ghi": 1, "cloud_cover": -1, "slope": -1, "ndvi": -1,
    "lc_forest": -1, "lc_water": -1, "lc_wetland": -1, "lc_snow": -1, "lc_urban": -1,
}
EXCLUDE_IGBP = {11: "permanent wetland", 13: "urban / built-up land", 15: "snow and ice", 17: "water"}
MAX_SLOPE = 15.0

SANITY_SITES = [
    # name, lat, lon, expectation
    ("Bhadla Solar Park, India", 27.5397, 71.9157, "High"),
    ("Atacama Desert, Chile", -23.50, -69.20, "High"),
    ("Kamuthi Solar Park, India", 9.35, 78.38, "High"),
    ("Central Sahara, Algeria", 23.00, 10.00, "High"),
    ("Nevada desert (Tonopah), USA", 38.23, -117.36, "High"),
    ("Benban Solar Park, Egypt", 24.45, 32.73, "High"),
    ("Tengger Desert Solar Park, China", 37.55, 105.05, "High"),
    ("Noor Ouarzazate, Morocco", 31.05, -6.87, "High"),
    ("Berlin city centre, Germany", 52.52, 13.40, "Low"),
    ("Amazon rainforest, Brazil", -3.50, -62.00, "Low"),
    ("Siberian taiga, Russia", 60.00, 100.00, "Low"),
    ("Greenland ice sheet", 71.00, -40.00, "Low"),
]


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def physical_features(df: pd.DataFrame, epoch: str) -> pd.DataFrame:
    return engineer_global(df, epoch=epoch).drop(columns=HUMAN_COLS)


def region_class_weights(region: pd.Series, y: np.ndarray) -> np.ndarray:
    """Each (region, class) cell gets the same total weight; capped."""
    df = pd.DataFrame({"r": pd.Series(region).fillna("Other").to_numpy(), "y": y})
    counts = df.groupby(["r", "y"]).size()
    target = len(df) / len(counts)
    w = np.array([target / counts[(r, c)] for r, c in zip(df["r"], df["y"])], float)
    w = np.minimum(w, np.median(w) * MAX_WEIGHT_RATIO)
    return w * len(w) / w.sum()


def physical_plausibility(df: pd.DataFrame, epoch: str = "pre") -> np.ndarray:
    """AHP physical overlay (GHI, slope, land cover, NDVI, air temperature;
    accessibility removed, weights renormalised) with exclusions, 0-100."""
    from .ahp import GLOBAL_CRITERIA, global_scores

    sfx = "_pre" if epoch == "pre" else ""
    w = json.loads((config.PROCESSED_DIR / "ahp_results.json").read_text())["global"]["weights"]
    crit = [c for c in GLOBAL_CRITERIA if c != "Accessibility"]
    ww = np.array([w[c] for c in crit])
    ww = ww / ww.sum()
    d = {"ghi": df["ghi"], "slope": df["slope"], "landcover": df[f"landcover{sfx}"],
         "accessibility": df["accessibility"], "ndvi": df[f"ndvi{sfx}"], "temperature": df["temperature"]}
    r = global_scores({k: v.to_numpy(float) for k, v in d.items()})
    total = sum(wi * np.nan_to_num(r["scores"][c], nan=0.5) for c, wi in zip(crit, ww))
    return 100 * total * r["constraint"]


def make_model(features: list[str]) -> Pipeline:
    cst = tuple(MONOTONE.get(f, 0) for f in features)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, min_child_weight=10, reg_lambda=5.0,
                                monotone_constraints=cst, tree_method="hist", eval_metric="logloss",
                                n_jobs=-1, random_state=config.RANDOM_SEED)),
    ])


def temper(p, T: float = SCORE_TEMPERATURE) -> np.ndarray:
    """0-100 score from a probability via logit temperature scaling."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return 100.0 / (1.0 + np.exp(-np.log(p / (1 - p)) / T))


def score_frame(pipeline, df: pd.DataFrame, features: list[str], epoch: str = "current") -> np.ndarray:
    """Physical suitability 0-100 for rows of raw features, exclusions applied."""
    X = physical_features(df, epoch)[features]
    s = temper(pipeline.predict_proba(X)[:, 1])
    lc = df["landcover"] if epoch == "current" else df["landcover_pre"]
    excl = np.array([exclusion_reason(a, b) is not None for a, b in zip(lc, df["slope"])])
    s[excl] = 0.0
    return s


EXCLUSION_MESSAGES = {
    "urban / built-up land": "Excluded: urban area, not suitable for ground-mounted solar.",
    "snow and ice": "Excluded: permanent snow or ice, not suitable for ground-mounted solar.",
    "water": "Excluded: water body, not suitable for ground-mounted solar.",
    "permanent wetland": "Excluded: wetland, protected and not suitable for ground-mounted solar.",
}


def exclusion_message(reason: str | None) -> str | None:
    """Plain-language explanation shown to users when a site is excluded."""
    if not reason:
        return None
    if reason.startswith("slope"):
        return f"Excluded: terrain steeper than {MAX_SLOPE:.0f}°, too steep for ground-mounted solar."
    return EXCLUSION_MESSAGES.get(reason, f"Excluded: {reason}, not suitable for ground-mounted solar.")


def exclusion_reason(landcover, slope) -> str | None:
    if landcover is not None and not pd.isna(landcover) and int(round(landcover)) in EXCLUDE_IGBP:
        return EXCLUDE_IGBP[int(round(landcover))]
    if slope is not None and not pd.isna(slope) and slope > MAX_SLOPE:
        return f"slope above {MAX_SLOPE:.0f}°"
    return None


def region_mean_auc(y, p, region) -> float:
    aucs = []
    for r in pd.unique(region):
        m = region == r
        if len(np.unique(y[m])) == 2:
            aucs.append(roc_auc_score(y[m], p[m]))
    return float(np.mean(aucs))


def cls(s: float) -> str:
    return "High" if s >= 70 else ("Medium" if s >= 40 else "Low")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(pipe_factory, X, y, w, region, groups, tr, te, train_mask=None, reliable=None) -> dict:
    """Fit on the training split (optionally restricted to `train_mask`) and report
    AUC on the FULL test split and on plants vs reliable negatives only."""
    train_mask = np.ones(len(y), bool) if train_mask is None else train_mask
    reliable = np.ones(len(y), bool) if reliable is None else reliable
    tr_ = tr[train_mask[tr]]
    pipe = pipe_factory()
    pipe.fit(X.iloc[tr_], y[tr_], model__sample_weight=None if w is None else w[tr_])
    p = pipe.predict_proba(X.iloc[te])[:, 1]
    rel_te = reliable[te] | (y[te] == 1)
    spatial, spatial_rel = [], []
    for a, b in GroupKFold(n_splits=5).split(X, y, groups):
        a = a[train_mask[a]]
        m = pipe_factory()
        m.fit(X.iloc[a], y[a], model__sample_weight=None if w is None else w[a])
        pb = m.predict_proba(X.iloc[b])[:, 1]
        spatial.append(roc_auc_score(y[b], pb))
        rb = reliable[b] | (y[b] == 1)
        spatial_rel.append(roc_auc_score(y[b][rb], pb[rb]))
    return {"n_train": int(len(tr_)),
            "test_auc": float(roc_auc_score(y[te], p)),
            "test_auc_reliable": float(roc_auc_score(y[te][rel_te], p[rel_te])),
            "test_auc_region_mean": region_mean_auc(y[te], p, region[te]),
            "spatial_cv_auc": float(np.mean(spatial)), "spatial_cv_std": float(np.std(spatial)),
            "spatial_cv_auc_reliable": float(np.mean(spatial_rel)),
            "_pipe": pipe}


def sanity_table(models: dict[str, Pipeline], features: list[str], apply_mask: dict[str, bool]) -> pd.DataFrame:
    """Score each sanity site. Uses one cached Earth Engine sample at the exact
    coordinates when available, else the nearest cached 0.5 deg grid cell."""
    from .predict import Predictor

    P = Predictor(enable_live=False)
    exact = pd.read_csv(SANITY_FEATURES_CSV).set_index("site") if SANITY_FEATURES_CSV.exists() else None
    rows = []
    for name, lat, lon, expect in SANITY_SITES:
        if exact is not None and name in exact.index:
            cell, dist = exact.loc[name], 0.0
        else:
            cell, dist = P.nearest_cell(lat, lon)
        raw = pd.DataFrame([{**cell.to_dict(), "lat": lat, "lon": lon}])
        X = physical_features(raw, "current")[features]
        row = {"site": name, "expected": expect, "feature_km": round(dist, 1),
               "GHI": round(float(cell["ghi"]), 2),
               "IGBP": int(cell["landcover"]) if not pd.isna(cell["landcover"]) else None}
        reason = exclusion_reason(cell["landcover"], cell["slope"])
        for label, m in models.items():
            p = float(m.predict_proba(X)[0, 1])
            s = float(temper(p)) if label.startswith("final") else p * 100
            if apply_mask.get(label) and reason:
                s = 0.0
            row[label] = round(s, 1)
        row["excluded"] = reason or ""
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> dict:
    from .train import make_models, split_indices

    g = pd.read_csv(config.GLOBAL_DATASET_CSV)
    y = g["label"].to_numpy()
    region = g["region"].fillna("Other").to_numpy()
    groups = g["group_id"].to_numpy()
    X = physical_features(g, "pre")
    features = list(X.columns)
    tr, te = split_indices(y, g["pid"].to_numpy())

    plaus = physical_plausibility(g, "pre")
    reliable = (y == 0) & (plaus < PU_THRESHOLD)
    pu_mask = (y == 1) | reliable
    w_all = region_class_weights(g["region"], y)
    w_pu = np.zeros(len(y))
    w_pu[pu_mask] = region_class_weights(g["region"][pu_mask], y[pu_mask])
    print(f"[physical] PU: {reliable.sum():,} reliable negatives kept, "
          f"{int(((y == 0) & ~reliable).sum()):,} plausible negatives treated as unlabelled")

    res = {
        "before": evaluate(lambda: make_models()["Random Forest"], X, y, None, region, groups, tr, te,
                           reliable=reliable),
        "balanced": evaluate(lambda: make_model(features), X, y, w_all, region, groups, tr, te, reliable=reliable),
        "balanced_pu": evaluate(lambda: make_model(features), X, y, w_pu, region, groups, tr, te,
                                train_mask=pu_mask, reliable=reliable),
    }
    names = {"before": "BEFORE  RF, unweighted, all negatives",
             "balanced": "STEP 1  XGB, region-balanced + monotonic",
             "balanced_pu": "STEP 2  + PU reliable negatives (final)"}
    for k, r in res.items():
        print(f"[physical] {names[k]:42s} test AUC {r['test_auc']:.3f} | vs reliable neg. "
              f"{r['test_auc_reliable']:.3f} | region-mean {r['test_auc_region_mean']:.3f} | "
              f"spatial CV {r['spatial_cv_auc']:.3f} (reliable {r['spatial_cv_auc_reliable']:.3f})")

    final = make_model(features)
    final.fit(X[pu_mask], y[pu_mask], model__sample_weight=w_pu[pu_mask])
    before_full = make_models()["Random Forest"].fit(X, y)
    step1 = make_model(features).fit(X, y, model__sample_weight=w_all)

    table = sanity_table({"before": before_full, "step 1": step1, "final": final, "final + excl.": final},
                         features, {"final + excl.": True})
    table["verdict"] = ["✓" if cls(v) == e else "✗" for v, e in zip(table["final + excl."], table["expected"])]
    pd.set_option("display.width", 220)
    print(table.to_string(index=False))

    joblib.dump({"pipeline": final, "features": features, "model_name": "XGBoost (region-balanced, monotonic, PU)",
                 "variant": "physical_only_balanced_pu", "monotone_constraints": MONOTONE,
                 "pu_threshold": PU_THRESHOLD, "score_temperature": SCORE_TEMPERATURE, "exclusions": {"igbp": EXCLUDE_IGBP, "max_slope_deg": MAX_SLOPE},
                 "class_thresholds": {"High": 70, "Medium": 40}, "trained_rows": int(pu_mask.sum())},
                PHYSICAL_MODEL_PKL)
    out = {**{k: {kk: vv for kk, vv in r.items() if not kk.startswith("_")} for k, r in res.items()},
           "n_reliable_negatives": int(reliable.sum()), "n_unlabelled_negatives": int(((y == 0) & ~reliable).sum()),
           "method": {"weights": "equal total weight per (world region x class), capped at 10x median",
                      "pu": f"negatives with physical plausibility >= {PU_THRESHOLD:.0f} treated as unlabelled",
                      "score": f"sigmoid(logit(p) / {SCORE_TEMPERATURE}) x 100, exclusions -> 0",
                      "monotone_constraints": MONOTONE,
                      "exclusions": list(EXCLUDE_IGBP.values()) + [f"slope > {MAX_SLOPE:.0f}°"],
                      "features": features},
           "sanity": table.to_dict(orient="records")}
    out["heatmap"] = physical_heatmap(final, features)
    PHYSICAL_JSON.write_text(json.dumps(out, indent=2, default=str))
    return out


def physical_heatmap(pipeline, features: list[str]) -> dict:
    """Global 0.5 deg suitability map (headline layer) + combined heatmap JSON
    carrying all three scores per cell (suitability, development, AHP)."""
    import matplotlib.pyplot as plt

    from . import viz
    from .global_grid import grid_features_path

    viz.apply_style()
    g = pd.read_csv(grid_features_path(0.5))
    g["suitability"] = score_frame(pipeline, g, features)
    dev = pd.read_csv(config.PROCESSED_DIR / "global_grid_predictions_0p5.csv.gz", usecols=["pid", "score"])
    ahp = pd.read_csv(config.PROCESSED_DIR / "global_ahp_grid.csv", usecols=["pid", "ahp_score"])
    g = g.merge(dev.rename(columns={"score": "development"}), on="pid", how="left").merge(ahp, on="pid", how="left")
    g.to_csv(config.PROCESSED_DIR / "global_grid_scores_0p5.csv.gz", index=False, compression="gzip")

    step, lat0, lon0 = 0.5, config.GLOBAL_LAT_RANGE[0], -180.0
    nrows = int((config.GLOBAL_LAT_RANGE[1] - lat0) / step)
    grid = viz.points_to_grid(g["lat"].to_numpy(), g["lon"].to_numpy(), g["suitability"].to_numpy(),
                              lat0, lon0, step, nrows, 720)
    merc, bounds = viz.grid_to_mercator(grid, lat0, lon0, step, width=2048)
    viz.save_png(viz.colorize(merc), config.PROCESSED_DIR / "layers" / "physical_global.png")

    payload = {"step": step, "bounds": bounds, "n": int(len(g)),
               "lat": g["lat"].round(3).tolist(), "lon": g["lon"].round(3).tolist(),
               "suitability": g["suitability"].round(1).tolist(),
               "development": g["development"].round(1).tolist(),
               "ahp": g["ahp_score"].round(1).tolist()}
    (config.PROCESSED_DIR / "global_heatmap_0p5.json").write_text(json.dumps(payload, separators=(",", ":")))

    fig, ax = plt.subplots(figsize=(15, 6.2))
    ax.grid(False)
    ax.set_facecolor("#eef2f6")
    ax.imshow(viz.colorize(merc, alpha_ramp=None), interpolation="nearest",
              extent=[viz.merc_x(bounds["west"]), viz.merc_x(bounds["east"]),
                      viz.merc_y(bounds["south"]), viz.merc_y(bounds["north"])])
    ticks = np.array([-40, -20, 0, 20, 40, 60])
    ax.set_yticks(viz.merc_y(ticks), [f"{abs(t)}°{'N' if t > 0 else 'S'}" if t else "0°" for t in ticks])
    ax.set_xticks(viz.merc_x(np.array([-120, -60, 0, 60, 120])), ["120°W", "60°W", "0°", "60°E", "120°E"])
    viz.legend_bar(ax, label="Suitability (0-100)")
    ax.set_title("Physical suitability - global 0.5° grid (region-balanced PU model, exclusions applied)")
    plt.savefig(config.FIGURES_DIR / "suitability_map_global.png", bbox_inches="tight")
    plt.close(fig)
    s_ = g["suitability"]
    return {"mean": float(s_.mean()), "high_pct": float((s_ >= 70).mean() * 100),
            "medium_pct": float(((s_ >= 40) & (s_ < 70)).mean() * 100),
            "by_region": g.groupby("region")["suitability"].mean().round(1).to_dict()}


if __name__ == "__main__":
    main()
