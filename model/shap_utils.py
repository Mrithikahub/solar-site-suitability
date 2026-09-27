"""
SHAP (SHapley Additive exPlanations) helpers.

Exact Shapley values for tree ensembles: for one prediction
f(x) = base_value + sum_i phi_i, each phi_i is feature i's contribution.
For XGBoost these are in log-odds; for scikit-learn Random Forests they are in
probability units. `explain_row` also returns a probability-scaled version so
contributions can be shown as "percentage points of suitability" in the UI.

XGBoost models are explained with XGBoost's built-in TreeSHAP
(`Booster.predict(pred_contribs=True)`), which gives the same values as
`shap.TreeExplainer` but needs no extra library. That keeps the `shap` package
(and its numba / llvmlite stack) out of the API process; it is imported lazily
only for other model types (e.g. the Random Forests in the training notebooks).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class XGBExplainer:
    """TreeSHAP via XGBoost itself (no `shap` import)."""

    def __init__(self, est):
        self.booster = est.get_booster()

    def contributions(self, Xt: pd.DataFrame) -> tuple[np.ndarray, float]:
        import xgboost as xgb

        contrib = self.booster.predict(xgb.DMatrix(Xt), pred_contribs=True)   # (n, p + 1), last column = bias
        return np.asarray(contrib[:, :-1]), float(contrib[0, -1])


def make_explainer(pipeline):
    """Explainer for the final estimator of an (imputer -> model) pipeline."""
    est = pipeline.named_steps["model"]
    if hasattr(est, "get_booster"):
        return XGBExplainer(est)
    try:
        import shap
    except ImportError as e:  # pragma: no cover - only non-XGBoost models need the shap package
        raise ImportError("Explaining non-XGBoost models needs the `shap` package (pip install shap).") from e
    return shap.TreeExplainer(est)


def _transform(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    imp = pipeline.named_steps.get("impute")
    if imp is None:
        return X
    return pd.DataFrame(imp.transform(X), columns=X.columns, index=X.index)


def shap_values(explainer, pipeline, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """(n, p) SHAP matrix for the positive class and its base value."""
    Xt = _transform(pipeline, X)
    if isinstance(explainer, XGBExplainer):
        return explainer.contributions(Xt)
    sv = explainer.shap_values(Xt)
    base = explainer.expected_value
    if isinstance(sv, list):                  # older sklearn API: one array per class
        sv, base = sv[1], base[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                          # (n, p, classes)
        sv, base = sv[..., 1], np.atleast_1d(base)[1]
    return sv, float(np.atleast_1d(base)[0])


def explain_row(explainer, pipeline, x: pd.DataFrame, meta: dict, raw: dict | None = None,
                top: int | None = None, transform=None) -> dict:
    """Per-prediction explanation for a single-row DataFrame.

    Returns base value, model output, and per-feature contributions sorted by
    |impact|, each with its feature value, raw SHAP value and a probability-
    scaled contribution in percentage points (sums to score - base_score).
    """
    sv, base = shap_values(explainer, pipeline, x)
    phi = sv[0]
    est = pipeline.named_steps["model"]
    prob = float(pipeline.predict_proba(x)[0, 1])
    log_odds = hasattr(est, "get_booster")
    base_prob = float(1 / (1 + np.exp(-base))) if log_odds else base
    if transform is not None:          # explain a transformed score (e.g. temperature-scaled)
        prob, base_prob = transform(prob), transform(base_prob)
    total = phi.sum()
    scale = (prob - base_prob) / total if abs(total) > 1e-9 else 0.0
    items = []
    for name, v, p in zip(x.columns, x.iloc[0].to_numpy(), phi):
        m = meta.get(name, {"label": name, "unit": "", "source": ""})
        items.append({
            "feature": name, "label": m["label"], "unit": m["unit"], "source": m["source"],
            "value": None if pd.isna(v) else float(v),
            "shap": float(p), "contribution_pct": float(p * scale * 100),
        })
    items.sort(key=lambda d: -abs(d["shap"]))
    if top:
        items = items[:top]
    return {"base_score": base_prob * 100, "score": prob * 100, "units": "log-odds" if log_odds else "probability",
            "contributions": items}
