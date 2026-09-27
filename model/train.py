"""
Phase 3 - Machine-learning suitability models.

    python -m model.train

For BOTH the global model and the Tamil Nadu (local) model:
  * Random Forest and XGBoost, each wrapped as (median imputer -> model)
  * stratified 80/20 train/test split (grouped by site for Tamil Nadu so no
    solar park contributes points to both sides)
  * stratified 5-fold CV on the training set + spatially blocked 5-fold CV
    (GroupKFold over 2 deg blocks / sites) on all data
  * accuracy, precision, recall, F1, ROC-AUC, confusion matrix
  * label-leakage experiment:
        clean  = time-varying predictors from BEFORE construction
        leaky  = the same predictors measured NOW (they "see" the panels)
        as-specified = leaky + post-construction ESA WorldCover
    each evaluated on its own test set AND on pre-construction land, which is
    what a real prospective site looks like
  * physical-only ablation (drops human / grid-access predictors) to separate
    "physically suitable" from "resembles where developers actually built"
  * ML vs AHP on identical test points
  * SHAP global summary + feature importance
  * global 0.5 deg heatmap and Tamil Nadu ~285 m heatmap from the clean model
  * global model vs local model evaluated on the Tamil Nadu test points

Outputs: model/model.pkl, model/model_tn.pkl, data/processed/metrics.json,
data/processed/*heatmap*, report/figures/*.png
"""
from __future__ import annotations

import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config, viz
from .features import FEATURE_META, GLOBAL_FEATURE_META, engineer, engineer_global
from .shap_utils import make_explainer, shap_values

METRICS_JSON = config.PROCESSED_DIR / "metrics.json"
GLOBAL_MODEL_PKL = config.MODEL_DIR / "model.pkl"
TN_MODEL_PKL = config.MODEL_DIR / "model_tn.pkl"
LOW_SAMPLE_POSITIVES = 150        # regions with fewer labelled plants are flagged
CLASS_THRESHOLDS = {"High": 70, "Medium": 40}
SEED = config.RANDOM_SEED


# --------------------------------------------------------------------------- #
# Models + metrics
# --------------------------------------------------------------------------- #
def make_models() -> dict[str, Pipeline]:
    return {
        "Random Forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features="sqrt",
                                             class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
        ]),
        "XGBoost": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.8, min_child_weight=2, reg_lambda=1.0,
                                    tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=SEED)),
        ]),
    }


def metrics(y, p, thr: float = 0.5) -> dict:
    yhat = (np.asarray(p) >= thr).astype(int)
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0), "f1": f1_score(y, yhat, zero_division=0),
        "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
        "confusion_matrix": cm.tolist(), "n": int(len(y)), "n_pos": int(np.sum(y)),
    }


def rounded(d):
    if isinstance(d, dict):
        return {k: rounded(v) for k, v in d.items()}
    if isinstance(d, list):
        return [rounded(v) for v in d]
    if isinstance(d, (float, np.floating)):
        return None if np.isnan(d) else round(float(d), 4)
    if isinstance(d, (np.integer,)):
        return int(d)
    return d


def split_indices(y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stratified (and grouped) 80/20 split: fold 0 of StratifiedGroupKFold."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, te = next(sgkf.split(np.zeros(len(y)), y, groups))
    return tr, te


def cv_scores(pipe: Pipeline, X, y, groups=None) -> dict:
    """Stratified 5-fold AUC (train) or spatial GroupKFold AUC (all data)."""
    if groups is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        s = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    else:
        s = cross_val_score(pipe, X, y, cv=GroupKFold(n_splits=5), groups=groups, scoring="roc_auc", n_jobs=1)
    return {"mean": float(s.mean()), "std": float(s.std()), "folds": [float(v) for v in s]}


def oof_predictions(pipe: Pipeline, X, y, groups) -> np.ndarray:
    """Out-of-fold probabilities from spatially blocked CV."""
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = make_models()[pipe_name(pipe)]
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return oof


def pipe_name(pipe: Pipeline) -> str:
    return "XGBoost" if isinstance(pipe.named_steps["model"], XGBClassifier) else "Random Forest"


def score_class(score: float) -> str:
    if score >= CLASS_THRESHOLDS["High"]:
        return "High"
    if score >= CLASS_THRESHOLDS["Medium"]:
        return "Medium"
    return "Low"


# --------------------------------------------------------------------------- #
# One full experiment (global or Tamil Nadu)
# --------------------------------------------------------------------------- #
def run_experiment(name: str, variants: dict[str, pd.DataFrame], y: np.ndarray, split_groups: np.ndarray,
                   spatial_groups: np.ndarray, ahp: np.ndarray | None) -> dict:
    tr, te = split_indices(y, split_groups)
    out = {"n_total": int(len(y)), "n_pos": int(y.sum()), "n_train": int(len(tr)), "n_test": int(len(te)),
           "variants": {}, "split": {"train_idx": tr, "test_idx": te}}
    fitted = {}
    for vname, X in variants.items():
        out["variants"][vname] = {"features": list(X.columns), "models": {}}
        for mname, pipe in make_models().items():
            t0 = time.time()
            pipe.fit(X.iloc[tr], y[tr])
            p = pipe.predict_proba(X.iloc[te])[:, 1]
            res = {"test": metrics(y[te], p),
                   "cv_stratified_auc": cv_scores(make_models()[mname], X.iloc[tr], y[tr]),
                   "cv_spatial_auc": cv_scores(make_models()[mname], X, y, groups=spatial_groups)}
            # evaluate on pre-construction land (what a genuinely new site looks like)
            if vname != "as_specified":
                p_pre = pipe.predict_proba(variants["clean"].iloc[te][X.columns])[:, 1]
                res["test_on_preconstruction_land"] = metrics(y[te], p_pre)
            res["fit_seconds"] = round(time.time() - t0, 1)
            out["variants"][vname]["models"][mname] = res
            fitted[(vname, mname)] = (pipe, p)
            print(f"[{name}] {vname:12s} {mname:13s} test AUC {res['test']['roc_auc']:.3f}  "
                  f"F1 {res['test']['f1']:.3f}  CV {res['cv_stratified_auc']['mean']:.3f}  "
                  f"spatial CV {res['cv_spatial_auc']['mean']:.3f}"
                  + (f"  on pre-constr. land {res['test_on_preconstruction_land']['roc_auc']:.3f}"
                     if "test_on_preconstruction_land" in res else ""))
    # deployed = clean variant, best spatial-CV AUC
    best = max(("Random Forest", "XGBoost"),
               key=lambda m: out["variants"]["clean"]["models"][m]["cv_spatial_auc"]["mean"])
    out["deployed_model"] = best
    if ahp is not None:
        out["ahp_test"] = metrics(y[te], np.nan_to_num(ahp[te]) / 100.0)
        print(f"[{name}] AHP baseline test AUC {out['ahp_test']['roc_auc']:.3f}  F1 {out['ahp_test']['f1']:.3f}")
    out["_fitted"] = fitted
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_roc(res: dict, y_te: np.ndarray, ahp_te, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=viz.AXIS, lw=1, ls="--")
    curves = []
    if ahp_te is not None:
        curves.append(("AHP", np.nan_to_num(ahp_te) / 100))
    for m in ("Random Forest", "XGBoost"):
        curves.append((m, res["_fitted"][("clean", m)][1]))
    for label, p in curves:
        fpr, tpr, _ = roc_curve(y_te, p)
        auc = roc_auc_score(y_te, p)
        ax.plot(fpr, tpr, lw=2, color=viz.MODEL_COLORS[label], label=f"{label}  (AUC {auc:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_confusion(cm: list, title: str, path) -> None:
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    ax.grid(False)
    ax.imshow(cm, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "b", ["#f0f4fa", "#2a78d6"]))
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{cm[i, j] / cm[i].sum():.1%}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.6 else viz.INK, fontsize=10)
    ax.set_xticks([0, 1], ["Not suitable", "Suitable"])
    ax.set_yticks([0, 1], ["No plant", "Plant"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_importance(pipe: Pipeline, features: list[str], meta: dict, title: str, path, top: int = 15) -> list:
    est = pipe.named_steps["model"]
    imp = np.asarray(est.feature_importances_, float)
    imp = imp / imp.sum()
    order = np.argsort(imp)[::-1][:top]
    labels = [meta.get(features[i], {"label": features[i]})["label"] for i in order]
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(order) + 1.2))
    ax.barh(labels[::-1], imp[order][::-1] * 100, color=viz.SERIES[0], height=0.62)
    for yy, v in enumerate(imp[order][::-1] * 100):
        ax.text(v + 0.3, yy, f"{v:.1f}%", va="center", fontsize=8.5, color=viz.INK_2)
    ax.set_xlabel("Relative importance (%)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, imp[order].max() * 100 * 1.18)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return [{"feature": features[i], "label": meta.get(features[i], {"label": features[i]})["label"],
             "importance": float(imp[i])} for i in np.argsort(imp)[::-1]]


def fig_shap_summary(pipe: Pipeline, X: pd.DataFrame, meta: dict, title: str, path, n: int = 2000) -> list:
    Xs = X.sample(min(n, len(X)), random_state=SEED)
    expl = make_explainer(pipe)
    sv, _ = shap_values(expl, pipe, Xs)
    Xd = Xs.copy()
    Xd.columns = [meta.get(c, {"label": c})["label"] for c in Xs.columns]
    plt.figure()
    shap.summary_plot(sv, Xd, max_display=15, show=False, cmap=viz.SHAP_CMAP, plot_size=(8.5, 6.2))
    fig = plt.gcf()
    fig.patch.set_facecolor(viz.SURFACE)
    plt.title(title, loc="left", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close("all")
    mean_abs = np.abs(sv).mean(axis=0)
    return [{"feature": c, "label": meta.get(c, {"label": c})["label"], "mean_abs_shap": float(v)}
            for c, v in sorted(zip(Xs.columns, mean_abs), key=lambda t: -t[1])]


def fig_leakage(glob: dict, tn: dict, path) -> None:
    """Grouped bars: AUC on own test set vs AUC on pre-construction land."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (label, res) in zip(axes, (("Global model", glob), ("Tamil Nadu model", tn))):
        m = res["deployed_model"]
        groups = ["Own test set", "Pre-construction land"]
        x = np.arange(len(groups))
        for k, (variant, color) in enumerate((("clean", viz.VARIANT_COLORS["clean"]),
                                              ("leaky", viz.VARIANT_COLORS["leaky"]))):
            r = res["variants"][variant]["models"][m]
            vals = [r["test"]["roc_auc"], r["test_on_preconstruction_land"]["roc_auc"]]
            bars = ax.bar(x + (k - 0.5) * 0.36, vals, width=0.34, color=color,
                          label="Clean (pre-construction features)" if variant == "clean" else "Leaky (current features)")
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, color=viz.INK_2)
        ax.set_xticks(x, groups)
        ax.set_ylim(0.5, 1.04)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{label} ({m})")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_regions(rows: list[dict], path) -> None:
    rows = sorted(rows, key=lambda r: (r["auc"] if r["auc"] is not None else 0))
    fig, ax = plt.subplots(figsize=(8, 0.46 * len(rows) + 1.3))
    for i, r in enumerate(rows):
        v = r["auc"] or 0
        ax.barh(i, v, color=viz.SERIES[0], height=0.6, alpha=0.35 if r["low_sample"] else 1.0,
                hatch="///" if r["low_sample"] else None, edgecolor=viz.SERIES[0], lw=0)
        tag = "  ⚠ low sample" if r["low_sample"] else ""
        ax.text(v + 0.004, i, f"{v:.3f}  (n={r['n']:,}, plants={r['n_pos']:,}){tag}", va="center",
                fontsize=8.5, color=viz.INK_2)
    ax.set_yticks(range(len(rows)), [r["region"] for r in rows])
    ax.set_xlim(0.5, 1.12)
    ax.set_xlabel("ROC-AUC (spatial-block out-of-fold predictions)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Global model - performance by world region")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_model_comparison(res: dict, title: str, path) -> None:
    names = ["AHP", "Random Forest", "XGBoost"]
    mets = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    vals = {"AHP": res["ahp_test"]}
    for m in ("Random Forest", "XGBoost"):
        vals[m] = res["variants"]["clean"]["models"][m]["test"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(mets))
    for k, n in enumerate(names):
        v = [vals[n][m] for m in mets]
        bars = ax.bar(x + (k - 1) * 0.27, v, width=0.25, color=viz.MODEL_COLORS[n], label=n)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.01, f"{val:.2f}", ha="center", fontsize=7.5,
                    color=viz.INK_2)
    ax.set_xticks(x, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.set_ylim(0, 1.1)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper left")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Heatmaps
# --------------------------------------------------------------------------- #
def global_heatmap(pipe: Pipeline) -> dict:
    from .global_grid import grid_features_path

    g = pd.read_csv(grid_features_path(0.5))
    X = engineer_global(g, epoch="current")
    g["score"] = pipe.predict_proba(X)[:, 1] * 100
    g.to_csv(config.PROCESSED_DIR / "global_grid_predictions_0p5.csv.gz", index=False, compression="gzip")

    step, lat0, lon0 = 0.5, config.GLOBAL_LAT_RANGE[0], -180.0
    nrows = int((config.GLOBAL_LAT_RANGE[1] - lat0) / step)
    grid = viz.points_to_grid(g["lat"].to_numpy(), g["lon"].to_numpy(), g["score"].to_numpy(), lat0, lon0,
                              step, nrows, 720)
    merc, bounds = viz.grid_to_mercator(grid, lat0, lon0, step, width=2048)
    viz.save_png(viz.colorize(merc), config.PROCESSED_DIR / "layers" / "ml_global.png")

    # compact JSON for the API (/heatmap): parallel arrays, 1 decimal
    payload = {"step": step, "bounds": bounds, "n": int(len(g)),
               "lat": g["lat"].round(3).tolist(), "lon": g["lon"].round(3).tolist(),
               "score": g["score"].round(1).tolist()}
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
    viz.legend_bar(ax, label="ML suitability (0-100)")
    ax.set_title("Machine-learning suitability - global 0.5° grid (XGBoost / RF, current conditions)")
    plt.savefig(config.FIGURES_DIR / "ml_map_global.png", bbox_inches="tight")
    plt.close(fig)
    return {"mean": float(g["score"].mean()), "high_pct": float((g["score"] >= 70).mean() * 100),
            "by_region": g.groupby("region")["score"].mean().round(1).to_dict()}


def tn_heatmap(pipe: Pipeline, features: list[str]) -> dict:
    from .tn_rasters import load_stack

    s = load_stack()
    mask = s["mask"]
    cols = ["ghi", "temperature", "cloud_cover", "elevation", "slope", "aspect", "ndvi", "ndbi", "lst",
            "landcover", "dist_road_km", "dist_powerline_km", "dist_substation_km"]
    df = pd.DataFrame({c: s[c][mask] for c in cols})
    X = engineer(df, use_baseline=False, include_landcover=False)[features]
    score = np.full(mask.shape, np.nan, dtype=np.float32)
    proba = np.concatenate([pipe.predict_proba(X.iloc[i:i + 200_000])[:, 1] for i in range(0, len(X), 200_000)])
    score[mask] = proba * 100
    np.save(config.PROCESSED_DIR / "tn_ml_score.npy", score)
    viz.save_png(viz.colorize(score), config.PROCESSED_DIR / "layers" / "ml_tn.png")

    lats, lons = s["lats"], s["lons"]
    fig, ax = plt.subplots(figsize=(7.5, 9))
    ax.grid(False)
    ax.imshow(viz.colorize(score, alpha_ramp=None), interpolation="nearest",
              extent=[viz.merc_x(lons[0]), viz.merc_x(lons[-1]), viz.merc_y(lats[-1]), viz.merc_y(lats[0])])
    ticks = np.arange(8, 14.1, 1)
    ax.set_yticks(viz.merc_y(ticks), [f"{t:.0f}°N" for t in ticks])
    ax.set_xticks(viz.merc_x(np.arange(77, 80.1, 1)), [f"{t:.0f}°E" for t in np.arange(77, 80.1, 1)])
    viz.legend_bar(ax, label="ML suitability (0-100)")
    ax.set_title("ML suitability - Tamil Nadu (~285 m, clean model)")
    plt.savefig(config.FIGURES_DIR / "ml_map_tamil_nadu.png", bbox_inches="tight")
    plt.close(fig)
    v = score[mask]
    return {"mean": float(np.nanmean(v)), "high_pct": float(np.mean(v >= 70) * 100)}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> dict:
    viz.apply_style()
    report: dict = {"class_thresholds": CLASS_THRESHOLDS, "low_sample_positive_threshold": LOW_SAMPLE_POSITIVES}

    # ============================== GLOBAL ==================================
    g = pd.read_csv(config.GLOBAL_DATASET_CSV)
    ahp_g = pd.read_csv(config.PROCESSED_DIR / "global_ahp_points.csv")[["pid", "ahp_score"]]
    g = g.merge(ahp_g, on="pid", how="left")
    y = g["label"].to_numpy()
    variants = {
        "clean": engineer_global(g, epoch="pre"),
        "leaky": engineer_global(g, epoch="current"),
        "as_specified": engineer_global(g, epoch="current", landcover_source="worldcover"),
    }
    # physical-only ablation: no human / infrastructure proxies (lights, access, population)
    variants["physical_only"] = variants["clean"].drop(columns=["log_access", "log_lights", "log_pop"])
    G = run_experiment("global", variants, y, split_groups=g["pid"].to_numpy(),
                       spatial_groups=g["group_id"].to_numpy(), ahp=g["ahp_score"].to_numpy())
    te = G["split"]["test_idx"]
    best = G["deployed_model"]
    pipe_g, p_g = G["_fitted"][("clean", best)]
    Xc = variants["clean"]
    print(f"[global] deployed model: {best} (clean)")

    # per-region: spatial-block out-of-fold predictions for every point
    oof = oof_predictions(pipe_g, Xc, y, g["group_id"].to_numpy())
    region_rows = []
    pos_per_region = g[g.label == 1]["region"].value_counts()
    for reg, sub in g.assign(oof=oof).groupby("region"):
        n_pos_total = int(pos_per_region.get(reg, 0))
        m = metrics(sub["label"].to_numpy(), sub["oof"].to_numpy())
        te_sub = g.iloc[te][g.iloc[te]["region"] == reg]
        region_rows.append({
            "region": reg, "n": m["n"], "n_pos": m["n_pos"], "auc": m["roc_auc"], "f1": m["f1"],
            "precision": m["precision"], "recall": m["recall"], "accuracy": m["accuracy"],
            "n_test": int(len(te_sub)), "low_sample": n_pos_total < LOW_SAMPLE_POSITIVES,
            "note": (f"Only {n_pos_total} labelled plants - metrics are unstable; treat predictions in this "
                     f"region with caution." if n_pos_total < LOW_SAMPLE_POSITIVES else ""),
        })
    fig_regions([r for r in region_rows if r["auc"] is not None], config.FIGURES_DIR / "per_region_auc.png")

    fig_roc(G, y[te], g["ahp_score"].to_numpy()[te], "ROC - global test set", config.FIGURES_DIR / "roc_global.png")
    fig_confusion(G["variants"]["clean"]["models"][best]["test"]["confusion_matrix"],
                  f"Confusion matrix - global ({best})", config.FIGURES_DIR / "confusion_global.png")
    imp_g = fig_importance(pipe_g, list(Xc.columns), GLOBAL_FEATURE_META, f"Feature importance - global ({best})",
                           config.FIGURES_DIR / "feature_importance_global.png")
    shap_g = fig_shap_summary(pipe_g, Xc.iloc[te], GLOBAL_FEATURE_META, "SHAP summary - global model",
                              config.FIGURES_DIR / "shap_summary_global.png")
    fig_model_comparison(G, "AHP vs ML - global test set", config.FIGURES_DIR / "ahp_vs_ml_global.png")

    # refit deployed model on ALL data for serving
    final_g = make_models()[best].fit(Xc, y)
    heat = global_heatmap(final_g)
    joblib.dump({"pipeline": final_g, "features": list(Xc.columns), "model_name": best, "variant": "clean",
                 "epoch_train": config.GLOBAL_BASELINE, "epoch_predict": config.GLOBAL_CURRENT,
                 "class_thresholds": CLASS_THRESHOLDS, "trained_rows": int(len(y))}, GLOBAL_MODEL_PKL)

    # ============================ TAMIL NADU ================================
    t = pd.read_csv(config.DATASET_CSV)
    ahp_t = pd.read_csv(config.PROCESSED_DIR / "tn_ahp_points.csv")[["pid", "ahp_score"]]
    t = t.merge(ahp_t, on="pid", how="left")
    yt = t["label"].to_numpy()
    tv = {
        "clean": engineer(t, use_baseline=True, include_landcover=False),
        "leaky": engineer(t, use_baseline=False, include_landcover=False),
        "as_specified": engineer(t, use_baseline=False, include_landcover=True),
    }
    tv["physical_only"] = tv["clean"].drop(columns=["dist_road_km", "dist_powerline_km", "dist_substation_km"])
    T = run_experiment("tamil_nadu", tv, yt, split_groups=t["group_id"].to_numpy(),
                       spatial_groups=t["group_id"].to_numpy(), ahp=t["ahp_score"].to_numpy())
    tte = T["split"]["test_idx"]
    tbest = T["deployed_model"]
    pipe_t, p_t = T["_fitted"][("clean", tbest)]
    print(f"[tamil_nadu] deployed model: {tbest} (clean)")
    fig_roc(T, yt[tte], t["ahp_score"].to_numpy()[tte], "ROC - Tamil Nadu test set (grouped by site)",
            config.FIGURES_DIR / "roc_tamil_nadu.png")
    fig_confusion(T["variants"]["clean"]["models"][tbest]["test"]["confusion_matrix"],
                  f"Confusion matrix - Tamil Nadu ({tbest})", config.FIGURES_DIR / "confusion_tamil_nadu.png")
    imp_t = fig_importance(pipe_t, list(tv["clean"].columns), FEATURE_META, f"Feature importance - Tamil Nadu ({tbest})",
                           config.FIGURES_DIR / "feature_importance_tamil_nadu.png")
    shap_t = fig_shap_summary(pipe_t, tv["clean"].iloc[tte], FEATURE_META, "SHAP summary - Tamil Nadu model",
                              config.FIGURES_DIR / "shap_summary_tamil_nadu.png")
    fig_model_comparison(T, "AHP vs ML - Tamil Nadu test set", config.FIGURES_DIR / "ahp_vs_ml_tamil_nadu.png")
    fig_leakage(G, T, config.FIGURES_DIR / "leakage_experiment.png")

    # global model vs local model on the SAME Tamil Nadu test points
    tg = pd.read_csv(config.TN_GLOBAL_FEATURES_CSV)
    tg = t[["pid"]].merge(tg, on="pid", how="left")
    Xg_tn = engineer_global(tg, epoch="pre")[list(Xc.columns)]
    # pipe_g was fitted on the global training split only; the OSM Tamil Nadu points are a different
    # label source and never part of the global training data, so this is a true out-of-sample test.
    p_global_on_tn = pipe_g.predict_proba(Xg_tn.iloc[tte])[:, 1]
    gvl = {"local": T["variants"]["clean"]["models"][tbest]["test"],
           "global": metrics(yt[tte], p_global_on_tn),
           "ahp": T["ahp_test"],
           "note": "Evaluated on the Tamil Nadu site-grouped test split. The global model is trained on the "
                   "Kruitwagen inventory (independent labels, 2008-10 baseline, MODIS 500 m), the local model on "
                   "OSM Tamil Nadu plants (2013-15 Landsat baseline, 30 m + OSM grid distances)."}
    print(f"[tn] local {tbest} AUC {gvl['local']['roc_auc']:.3f} | global model on TN AUC {gvl['global']['roc_auc']:.3f}"
          f" | AHP AUC {gvl['ahp']['roc_auc']:.3f}")

    final_t = make_models()[tbest].fit(tv["clean"], yt)
    tn_heat = tn_heatmap(final_t, list(tv["clean"].columns))
    joblib.dump({"pipeline": final_t, "features": list(tv["clean"].columns), "model_name": tbest, "variant": "clean",
                 "epoch_train": {"spectral": f"{config.BASELINE_START}..{config.BASELINE_END} (Landsat 8)"},
                 "class_thresholds": CLASS_THRESHOLDS, "trained_rows": int(len(yt))}, TN_MODEL_PKL)

    # ============================== REPORT ==================================
    for R_ in (G, T):
        R_.pop("_fitted")
        R_.pop("split")
    report["global"] = {**G, "per_region": region_rows, "feature_importance": imp_g, "shap_importance": shap_g,
                        "heatmap": heat}
    report["tamil_nadu"] = {**T, "feature_importance": imp_t, "shap_importance": shap_t, "heatmap": tn_heat,
                            "global_vs_local": gvl}
    METRICS_JSON.write_text(json.dumps(rounded(report), indent=2))
    print(f"[done] metrics -> {METRICS_JSON}")

    # headline physical suitability model (region-balanced, PU, monotonic) + its heatmap
    from . import physical
    physical.main()
    return report


if __name__ == "__main__":
    main()
