"""
Shared plotting style + map rendering helpers.

Colour roles follow a validated palette:
  * categorical, fixed order: blue, orange, aqua (validated all-pairs for CVD)
  * suitability (sequential magnitude): single orange hue, light -> dark
  * SHAP feature value: diverging blue <-> red through a neutral grey
"""
from __future__ import annotations

import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]           # fixed order: slot 1, 2, 3
MODEL_COLORS = {"AHP": SERIES[0], "Random Forest": SERIES[1], "XGBoost": SERIES[2]}
VARIANT_COLORS = {"clean": SERIES[0], "leaky": SERIES[1]}

SUIT_STEPS = ["#fde7d9", "#f9c4a2", "#f39a6b", "#eb6834", "#c94f1f", "#9c3a14", "#6e280c"]
SUIT_CMAP = LinearSegmentedColormap.from_list("suitability", SUIT_STEPS)
SHAP_CMAP = LinearSegmentedColormap.from_list("shap_div", ["#2a78d6", "#f0efec", "#e34948"])


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200,
    })


# --------------------------------------------------------------------------- #
# Map rendering (Web-Mercator PNG overlays for Leaflet)
# --------------------------------------------------------------------------- #
R = 6378137.0


def merc_x(lon):
    return R * np.radians(lon)


def merc_y(lat):
    return R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def inv_merc_y(y):
    return np.degrees(2 * np.arctan(np.exp(y / R)) - np.pi / 2)


def colorize(values: np.ndarray, cmap=SUIT_CMAP, vmin: float = 0, vmax: float = 100,
             alpha_ramp: tuple[float, float] | None = (0.35, 0.92)) -> np.ndarray:
    """Float raster -> RGBA uint8. NaN becomes transparent. With alpha_ramp,
    opacity rises with the value so low-suitability land recedes into the basemap."""
    t = np.clip((values - vmin) / (vmax - vmin), 0, 1)
    rgba = cmap(np.nan_to_num(t))
    if alpha_ramp is not None:
        a0, a1 = alpha_ramp
        rgba[..., 3] = a0 + (a1 - a0) * np.nan_to_num(t)
    rgba[np.isnan(values), 3] = 0
    return (rgba * 255).astype(np.uint8)


def categorical_rgba(codes: np.ndarray, color_map: dict[int, str], alpha: float = 0.85) -> np.ndarray:
    out = np.zeros(codes.shape + (4,), dtype=np.uint8)
    for code, hexc in color_map.items():
        m = codes == code
        h = hexc.lstrip("#")
        out[m] = [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(alpha * 255)]
    return out


def grid_to_mercator(values: np.ndarray, lat0: float, lon0: float, step: float,
                     width: int = 2048, smooth: bool = True, categorical: bool = False) -> tuple[np.ndarray, dict]:
    """Regular lat/lon grid (row 0 = southernmost cell, lat0/lon0 = SW corner)
    -> Web-Mercator raster of `width` px. Continuous data is resampled with
    NaN-aware bilinear interpolation; categorical data uses nearest cell."""
    nrows, ncols = values.shape
    south, north = lat0, lat0 + nrows * step
    west, east = lon0, lon0 + ncols * step
    y0, y1 = merc_y(south), merc_y(north)
    x0, x1 = R * math.radians(west), R * math.radians(east)
    height = int(round(width * (y1 - y0) / (x1 - x0)))

    lon_px = west + (np.arange(width) + 0.5) / width * (east - west)
    lat_px = inv_merc_y(y1 - (np.arange(height) + 0.5) / height * (y1 - y0))
    fc = (lon_px - west) / step - 0.5          # fractional cell index (centres at .0)
    fr = (lat_px - south) / step - 0.5

    if categorical or not smooth:
        ci = np.clip(np.round(fc).astype(int), 0, ncols - 1)
        ri = np.clip(np.round(fr).astype(int), 0, nrows - 1)
        out = values[ri[:, None], ci[None, :]]
    else:
        c0 = np.clip(np.floor(fc).astype(int), 0, ncols - 1)
        r0 = np.clip(np.floor(fr).astype(int), 0, nrows - 1)
        c1 = np.clip(c0 + 1, 0, ncols - 1)
        r1 = np.clip(r0 + 1, 0, nrows - 1)
        wc = np.clip(fc - c0, 0, 1)[None, :]
        wr = np.clip(fr - r0, 0, 1)[:, None]
        v = np.nan_to_num(values)
        m = (~np.isnan(values)).astype(float)
        num = np.zeros((height, width))
        den = np.zeros((height, width))
        for rr, cc, w in ((r0, c0, (1 - wr) * (1 - wc)), (r0, c1, (1 - wr) * wc),
                          (r1, c0, wr * (1 - wc)), (r1, c1, wr * wc)):
            num += v[rr[:, None], cc[None, :]] * m[rr[:, None], cc[None, :]] * w
            den += m[rr[:, None], cc[None, :]] * w
        out = np.where(den > 0.25, num / np.maximum(den, 1e-9), np.nan)
        # keep the land/sea edge crisp: only where the nearest cell is valid
        ci = np.clip(np.round(fc).astype(int), 0, ncols - 1)
        ri = np.clip(np.round(fr).astype(int), 0, nrows - 1)
        out[np.isnan(values[ri[:, None], ci[None, :]])] = np.nan
    return out, {"south": south, "west": west, "north": north, "east": east}


def save_png(rgba: np.ndarray, path) -> None:
    Image.fromarray(rgba, "RGBA").save(path, optimize=True)


def points_to_grid(lat: np.ndarray, lon: np.ndarray, val: np.ndarray, lat0: float, lon0: float,
                   step: float, nrows: int, ncols: int) -> np.ndarray:
    grid = np.full((nrows, ncols), np.nan)
    r = np.floor((lat - lat0) / step).astype(int)
    c = np.floor((lon - lon0) / step).astype(int)
    ok = (r >= 0) & (r < nrows) & (c >= 0) & (c < ncols)
    grid[r[ok], c[ok]] = val[ok]
    return grid


def legend_bar(ax, cmap=SUIT_CMAP, label="Suitability score", vmin=0, vmax=100):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    cb = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(label, color=INK_2)
    cb.outline.set_visible(False)
    return cb
