"""
Phase 2 - Analytic Hierarchy Process (AHP) weighted-overlay baseline.

Classic GIS multi-criteria decision analysis (Saaty, 1980):

  1. Build a pairwise comparison matrix A (n x n) on Saaty's 1-9 scale, where
     a_ij says how much more important criterion i is than criterion j and
     a_ji = 1 / a_ij.
  2. Criterion weights w = principal (Perron) eigenvector of A, normalised to 1.
  3. Consistency:  lambda_max = principal eigenvalue
                   CI = (lambda_max - n) / (n - 1)
                   CR = CI / RI(n)       (RI = Saaty's random index)
     CR < 0.10 means the judgements are acceptably consistent.
  4. Standardise every criterion to [0, 1] with linear fuzzy membership
     functions (benefit: higher is better, cost: lower is better).
  5. Suitability = 100 * sum_i w_i * s_i, multiplied by a Boolean constraint
     mask (water, wetland, built-up, snow, slope > 15 deg -> 0).

Two versions are produced from cached data only:
  * Tamil Nadu (~285 m): NASA POWER + decoded S2/Landsat/SRTM/WorldCover + OSM
  * Global (0.5 deg grid): ERA5-Land + MODIS + Copernicus DEM + MAP accessibility

    python -m model.ahp
"""
from __future__ import annotations

import importlib
import json

import numpy as np
import pandas as pd

from . import config
from .features import IGBP_GROUPS


class _LazyModule:
    """Import a module on first attribute access. The API imports this file only
    for `global_scores` / `overlay`; matplotlib and the plotting helpers load
    only when a map or figure is actually drawn."""

    def __init__(self, name: str, package: str | None = None):
        self._name, self._package, self._mod = name, package, None

    def __getattr__(self, attr):
        if self._mod is None:
            self._mod = importlib.import_module(self._name, self._package)
        return getattr(self._mod, attr)


plt = _LazyModule("matplotlib.pyplot")
viz = _LazyModule(".viz", __package__)

AHP_JSON = config.PROCESSED_DIR / "ahp_results.json"
RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

# --------------------------------------------------------------------------- #
# Land-cover suitability look-ups (0 = unsuitable, 1 = ideal)
# --------------------------------------------------------------------------- #
WORLDCOVER_SCORE = {10: 0.15, 20: 0.80, 30: 0.90, 40: 0.55, 50: 0.0, 60: 1.0,
                    70: 0.0, 80: 0.0, 90: 0.0, 95: 0.0, 100: 0.5}
WORLDCOVER_EXCLUDE = {50, 70, 80, 90, 95}
IGBP_SCORE = {1: 0.15, 2: 0.15, 3: 0.15, 4: 0.15, 5: 0.15, 6: 0.70, 7: 0.90, 8: 0.50, 9: 0.80,
              10: 0.90, 11: 0.0, 12: 0.55, 13: 0.0, 14: 0.50, 15: 0.0, 16: 1.0, 17: 0.0}
IGBP_EXCLUDE = {11, 13, 15, 17}
MAX_SLOPE_DEG = 15.0


# --------------------------------------------------------------------------- #
# AHP core
# --------------------------------------------------------------------------- #
def matrix_from_upper(names: list[str], upper: dict[tuple[str, str], float]) -> np.ndarray:
    n = len(names)
    A = np.ones((n, n))
    idx = {k: i for i, k in enumerate(names)}
    for (a, b), v in upper.items():
        A[idx[a], idx[b]] = v
        A[idx[b], idx[a]] = 1.0 / v
    return A


def ahp_weights(A: np.ndarray) -> dict:
    n = A.shape[0]
    vals, vecs = np.linalg.eig(A)
    k = np.argmax(vals.real)
    lam = vals.real[k]
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    ci = (lam - n) / (n - 1)
    cr = ci / RI[n] if RI[n] > 0 else 0.0
    return {"weights": w, "lambda_max": float(lam), "CI": float(ci), "CR": float(cr), "RI": RI[n]}


def benefit(x, lo, hi):
    """0 at <= lo, 1 at >= hi, linear in between."""
    return np.clip((np.asarray(x, float) - lo) / (hi - lo), 0, 1)


def cost(x, lo, hi):
    """1 at <= lo, 0 at >= hi, linear in between."""
    return 1.0 - benefit(x, lo, hi)


def trapezoid(x, a, b, c, d):
    """0 below a, rises to 1 at b, flat to c, falls to 0 at d."""
    x = np.asarray(x, float)
    return np.clip(np.minimum((x - a) / (b - a), (d - x) / (d - c)), 0, 1)


def lookup(codes, table: dict, default: float = 0.0):
    codes = np.asarray(codes, float)
    out = np.full(codes.shape, default)
    for k, v in table.items():
        out[np.round(codes) == k] = v
    out[np.isnan(codes)] = np.nan
    return out


# --------------------------------------------------------------------------- #
# Criteria definitions
# --------------------------------------------------------------------------- #
TN_CRITERIA = ["GHI", "Slope", "Land cover", "Dist. substation", "Dist. road", "NDVI", "Air temperature"]
TN_UPPER = {
    ("GHI", "Slope"): 2, ("GHI", "Land cover"): 2, ("GHI", "Dist. substation"): 3, ("GHI", "Dist. road"): 5,
    ("GHI", "NDVI"): 4, ("GHI", "Air temperature"): 6,
    ("Slope", "Land cover"): 1, ("Slope", "Dist. substation"): 2, ("Slope", "Dist. road"): 4,
    ("Slope", "NDVI"): 3, ("Slope", "Air temperature"): 5,
    ("Land cover", "Dist. substation"): 2, ("Land cover", "Dist. road"): 4, ("Land cover", "NDVI"): 3,
    ("Land cover", "Air temperature"): 5,
    ("Dist. substation", "Dist. road"): 3, ("Dist. substation", "NDVI"): 2, ("Dist. substation", "Air temperature"): 4,
    ("Dist. road", "NDVI"): 0.5, ("Dist. road", "Air temperature"): 2,
    ("NDVI", "Air temperature"): 3,
}
TN_STANDARDISATION = {
    "GHI": "benefit, 4.5 -> 5.8 kWh/m²/day",
    "Slope": "cost, 0 -> 10°",
    "Land cover": "look-up (bare 1.0, grass 0.9, shrub 0.8, crop 0.55, tree 0.15)",
    "Dist. substation": "cost, 0 -> 20 km",
    "Dist. road": "cost, 0 -> 5 km",
    "NDVI": "cost, 0.1 -> 0.6",
    "Air temperature": "cost, 25 -> 35 °C",
}

GLOBAL_CRITERIA = ["GHI", "Slope", "Land cover", "Accessibility", "NDVI", "Air temperature"]
GLOBAL_UPPER = {
    ("GHI", "Slope"): 2, ("GHI", "Land cover"): 2, ("GHI", "Accessibility"): 3, ("GHI", "NDVI"): 4,
    ("GHI", "Air temperature"): 5,
    ("Slope", "Land cover"): 1, ("Slope", "Accessibility"): 2, ("Slope", "NDVI"): 3, ("Slope", "Air temperature"): 4,
    ("Land cover", "Accessibility"): 2, ("Land cover", "NDVI"): 3, ("Land cover", "Air temperature"): 4,
    ("Accessibility", "NDVI"): 2, ("Accessibility", "Air temperature"): 3,
    ("NDVI", "Air temperature"): 2,
}
GLOBAL_STANDARDISATION = {
    "GHI": "benefit, 2.5 -> 6.5 kWh/m²/day",
    "Slope": "cost, 0 -> 10°",
    "Land cover": "IGBP look-up (barren 1.0, grass/open shrub 0.9, savanna 0.8, crop 0.55, forest 0.15)",
    "Accessibility": "cost, 30 -> 360 min to nearest city",
    "NDVI": "cost, 0.1 -> 0.7",
    "Air temperature": "trapezoid: 0 at -10 °C, 1 between 10-25 °C, 0 at 40 °C",
}


def tn_scores(d: dict, landcover: str = "worldcover") -> dict:
    """Standardised criteria for Tamil Nadu. `d` maps column -> array.
    landcover='igbp' uses MODIS 2008 land cover (pre-construction evaluation)."""
    if landcover == "igbp":
        lc = lookup(d["landcover_igbp"], IGBP_SCORE)
        excl = np.isin(np.round(d["landcover_igbp"]), list(IGBP_EXCLUDE))
    else:
        lc = lookup(d["landcover"], WORLDCOVER_SCORE)
        excl = np.isin(np.round(d["landcover"]), list(WORLDCOVER_EXCLUDE))
    s = {
        "GHI": benefit(d["ghi"], 4.5, 5.8),
        "Slope": cost(d["slope"], 0, 10),
        "Land cover": lc,
        "Dist. substation": cost(d["dist_substation_km"], 0, 20),
        "Dist. road": cost(d["dist_road_km"], 0, 5),
        "NDVI": cost(d["ndvi"], 0.1, 0.6),
        "Air temperature": cost(d["temperature"], 25, 35),
    }
    constraint = ~(excl | (np.asarray(d["slope"], float) > MAX_SLOPE_DEG))
    return {"scores": s, "constraint": constraint}


def global_scores(d: dict) -> dict:
    s = {
        "GHI": benefit(d["ghi"], 2.5, 6.5),
        "Slope": cost(d["slope"], 0, 10),
        "Land cover": lookup(d["landcover"], IGBP_SCORE),
        "Accessibility": cost(d["accessibility"], 30, 360),
        "NDVI": cost(d["ndvi"], 0.1, 0.7),
        "Air temperature": trapezoid(d["temperature"], -10, 10, 25, 40),
    }
    excl = np.isin(np.round(np.asarray(d["landcover"], float)), list(IGBP_EXCLUDE))
    constraint = ~(excl | (np.asarray(d["slope"], float) > MAX_SLOPE_DEG))
    return {"scores": s, "constraint": constraint}


def overlay(scores: dict, constraint, names: list[str], weights: np.ndarray) -> np.ndarray:
    total = np.zeros_like(np.asarray(scores[names[0]], float))
    for n, w in zip(names, weights):
        total = total + w * np.nan_to_num(scores[n], nan=0.5)   # missing -> neutral
    return 100.0 * total * np.asarray(constraint, float)


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #
def run_tn(weights: np.ndarray) -> dict:
    from .tn_rasters import load_stack

    s = load_stack()
    mask = s["mask"]
    res = tn_scores(s)
    score = overlay(res["scores"], res["constraint"], TN_CRITERIA, weights)
    score[~mask] = np.nan
    np.save(config.PROCESSED_DIR / "tn_ahp_score.npy", score.astype(np.float32))

    # web overlay (same Mercator frame as the other TN layers)
    viz.save_png(viz.colorize(score), config.PROCESSED_DIR / "layers" / "ahp_tn.png")

    # AHP score at the labelled points, using PRE-construction land state
    pts = pd.read_csv(config.DATASET_CSV)
    tg = pd.read_csv(config.TN_GLOBAL_FEATURES_CSV)[["pid", "landcover_pre"]]
    pts = pts.merge(tg, on="pid", how="left")
    d = {c: pts[c].to_numpy() for c in ["ghi", "slope", "dist_substation_km", "dist_road_km", "temperature"]}
    d["ndvi"] = pts["ndvi_pre"].to_numpy()
    d["landcover_igbp"] = pts["landcover_pre"].to_numpy()
    r = tn_scores(d, landcover="igbp")
    pts["ahp_score"] = overlay(r["scores"], r["constraint"], TN_CRITERIA, weights)
    pts[["pid", "label", "ahp_score"]].to_csv(config.PROCESSED_DIR / "tn_ahp_points.csv", index=False)

    valid = score[mask]
    return {"map_mean": float(np.nanmean(valid)), "excluded_pct": float(np.mean(valid == 0) * 100),
            "high_pct": float(np.mean(valid >= 70) * 100), "score": score, "lats": s["lats"], "lons": s["lons"]}


def run_global(weights: np.ndarray) -> dict:
    from .global_grid import grid_features_path

    g = pd.read_csv(grid_features_path(0.5))
    res = global_scores({c: g[c].to_numpy() for c in ["ghi", "slope", "landcover", "accessibility", "ndvi", "temperature"]})
    g["ahp_score"] = overlay(res["scores"], res["constraint"], GLOBAL_CRITERIA, weights)
    g[["pid", "lat", "lon", "ahp_score"]].to_csv(config.PROCESSED_DIR / "global_ahp_grid.csv", index=False)

    step, lat0, lon0 = 0.5, config.GLOBAL_LAT_RANGE[0], -180.0
    nrows = int((config.GLOBAL_LAT_RANGE[1] - lat0) / step)
    grid = viz.points_to_grid(g["lat"].to_numpy(), g["lon"].to_numpy(), g["ahp_score"].to_numpy(),
                              lat0, lon0, step, nrows, 720)
    merc, bounds = viz.grid_to_mercator(grid, lat0, lon0, step, width=2048)
    viz.save_png(viz.colorize(merc), config.PROCESSED_DIR / "layers" / "ahp_global.png")

    # AHP at labelled global points (pre-construction epoch)
    pts = pd.read_csv(config.GLOBAL_DATASET_CSV)
    d = {"ghi": pts["ghi"], "slope": pts["slope"], "landcover": pts["landcover_pre"],
         "accessibility": pts["accessibility"], "ndvi": pts["ndvi_pre"], "temperature": pts["temperature"]}
    r = global_scores({k: v.to_numpy() for k, v in d.items()})
    pts["ahp_score"] = overlay(r["scores"], r["constraint"], GLOBAL_CRITERIA, weights)
    pts[["pid", "label", "region", "ahp_score"]].to_csv(config.PROCESSED_DIR / "global_ahp_points.csv", index=False)
    return {"grid": grid, "bounds": bounds, "merc": merc, "g": g,
            "map_mean": float(g["ahp_score"].mean()), "excluded_pct": float((g["ahp_score"] == 0).mean() * 100),
            "high_pct": float((g["ahp_score"] >= 70).mean() * 100)}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_matrix(A: np.ndarray, names: list[str], res: dict, title: str, path) -> None:
    n = len(names)
    fig, ax = plt.subplots(figsize=(1.05 * n + 3.2, 0.62 * n + 1.6))
    ax.grid(False)
    ax.imshow(np.log(A), cmap=viz.SHAP_CMAP, vmin=-np.log(9), vmax=np.log(9))
    for i in range(n):
        for j in range(n):
            v = A[i, j]
            txt = f"{v:.0f}" if v >= 1 and abs(v - round(v)) < 1e-9 else f"1/{1 / v:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=viz.INK)
    ax.set_xticks(range(n), names, rotation=35, ha="right")
    ax.set_yticks(range(n), names)
    ax.tick_params(length=0, colors=viz.INK_2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{title}\nλmax = {res['lambda_max']:.3f}   CI = {res['CI']:.3f}   CR = {res['CR']:.3f}"
                 f" ({'consistent' if res['CR'] < 0.1 else 'INCONSISTENT'})", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_weights(names: list[str], w: np.ndarray, title: str, path) -> None:
    order = np.argsort(w)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(names) + 1))
    ax.barh(np.array(names)[order], w[order] * 100, color=viz.SERIES[0], height=0.6)
    for y, v in enumerate(w[order] * 100):
        ax.text(v + 0.6, y, f"{v:.1f}%", va="center", fontsize=9, color=viz.INK_2)
    ax.set_xlabel("Weight (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.set_xlim(0, max(w) * 100 * 1.18)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_tn_map(tn: dict, path) -> None:
    score = tn["score"]
    lats, lons = tn["lats"], tn["lons"]
    fig, ax = plt.subplots(figsize=(7.5, 9))
    ax.grid(False)
    extent = [viz.merc_x(lons[0]), viz.merc_x(lons[-1]), viz.merc_y(lats[-1]), viz.merc_y(lats[0])]
    rgba = viz.colorize(score, alpha_ramp=None)
    ax.imshow(rgba, extent=extent, interpolation="nearest")
    ticks = np.arange(8, 14.1, 1)
    ax.set_yticks(viz.merc_y(ticks), [f"{t:.0f}°N" for t in ticks])
    ax.set_xticks(viz.merc_x(np.arange(77, 80.1, 1)), [f"{t:.0f}°E" for t in np.arange(77, 80.1, 1)])
    viz.legend_bar(ax, label="AHP suitability (0-100)")
    ax.set_title("AHP suitability - Tamil Nadu (~285 m)")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_global_map(gl: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(15, 6.2))
    ax.grid(False)
    b = gl["bounds"]
    rgba = viz.colorize(gl["merc"], alpha_ramp=None)
    ax.imshow(rgba, extent=[viz.merc_x(b["west"]), viz.merc_x(b["east"]), viz.merc_y(b["south"]), viz.merc_y(b["north"])],
              interpolation="nearest")
    ax.set_facecolor("#eef2f6")
    ticks = np.array([-40, -20, 0, 20, 40, 60])
    ax.set_yticks(viz.merc_y(ticks), [f"{abs(t)}°{'N' if t >= 0 else 'S'}" if t else "0°" for t in ticks])
    ax.set_xticks(viz.merc_x(np.array([-120, -60, 0, 60, 120])), ["120°W", "60°W", "0°", "60°E", "120°E"])
    viz.legend_bar(ax, label="AHP suitability (0-100)")
    ax.set_title("AHP suitability - global 0.5° grid")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> dict:
    viz.apply_style()
    out = {}
    for key, names, upper, std in (("tamil_nadu", TN_CRITERIA, TN_UPPER, TN_STANDARDISATION),
                                   ("global", GLOBAL_CRITERIA, GLOBAL_UPPER, GLOBAL_STANDARDISATION)):
        A = matrix_from_upper(names, upper)
        res = ahp_weights(A)
        out[key] = {"criteria": names, "matrix": A.round(4).tolist(),
                    "weights": {n: round(float(w), 4) for n, w in zip(names, res["weights"])},
                    "lambda_max": res["lambda_max"], "CI": res["CI"], "CR": res["CR"], "RI": res["RI"],
                    "standardisation": std, "max_slope_deg": MAX_SLOPE_DEG}
        label = "Tamil Nadu" if key == "tamil_nadu" else "Global"
        fig_matrix(A, names, res, f"AHP pairwise comparison matrix - {label}",
                   config.FIGURES_DIR / f"ahp_matrix_{key}.png")
        fig_weights(names, res["weights"], f"AHP criterion weights - {label}",
                    config.FIGURES_DIR / f"ahp_weights_{key}.png")
        print(f"[ahp] {label}: lambda_max={res['lambda_max']:.3f} CI={res['CI']:.4f} CR={res['CR']:.4f}")
        for n, w in sorted(zip(names, res["weights"]), key=lambda t: -t[1]):
            print(f"        {n:18s} {w * 100:5.1f}%")

    tn = run_tn(np.array(list(out["tamil_nadu"]["weights"].values())))
    fig_tn_map(tn, config.FIGURES_DIR / "ahp_map_tamil_nadu.png")
    gl = run_global(np.array(list(out["global"]["weights"].values())))
    fig_global_map(gl, config.FIGURES_DIR / "ahp_map_global.png")
    for key, r in (("tamil_nadu", tn), ("global", gl)):
        out[key]["map_summary"] = {"mean_score": round(r["map_mean"], 2), "excluded_pct": round(r["excluded_pct"], 2),
                                   "high_pct": round(r["high_pct"], 2)}
        print(f"[ahp] {key} map: mean {r['map_mean']:.1f}, excluded {r['excluded_pct']:.1f}%, "
              f"high (>=70) {r['high_pct']:.1f}%")
    AHP_JSON.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
