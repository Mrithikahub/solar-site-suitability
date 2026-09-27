"""
NASA POWER (Prediction Of Worldwide Energy Resources) climatology.

POWER distributes CERES/MERRA-2 derived surface solar and meteorological data.
Solar parameters are produced on a 1 deg x 1 deg grid and meteorology on a
0.5 deg x 0.625 deg grid, so querying every sample point individually would be
wasteful (thousands of calls returning identical numbers). Instead we:

  1. query a regular 0.5 deg lattice covering the region (~130 calls, cached),
  2. bilinearly interpolate the lattice to any point.

The same cached lattice is reused by the backend for live predictions, so the
API is never hit at request time.

Parameters (annual climatological mean):
  ALLSKY_SFC_SW_DWN  all-sky surface shortwave downward irradiance = GHI  [kWh/m^2/day]
  T2M                air temperature at 2 m                              [deg C]
  CLOUD_AMT          cloud amount                                        [%]
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from scipy.interpolate import RegularGridInterpolator

from . import config

POWER_URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"
POWER_PARAMS = {"ALLSKY_SFC_SW_DWN": "ghi", "T2M": "temperature", "CLOUD_AMT": "cloud_cover"}


def _fetch_point(lat: float, lon: float) -> dict:
    cache = config.CACHE_DIR / "power" / f"{lat:.2f}_{lon:.2f}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return json.loads(cache.read_text())

    params = {
        "parameters": ",".join(POWER_PARAMS),
        "community": "RE",
        "latitude": lat,
        "longitude": lon,
        "start": config.POWER_START_YEAR,
        "end": config.POWER_END_YEAR,
        "format": "JSON",
    }
    last = None
    for attempt in range(5):
        try:
            r = requests.get(POWER_URL, params=params, timeout=90)
            r.raise_for_status()
            data = r.json()["properties"]["parameter"]
            row = {"lat": lat, "lon": lon}
            for p, name in POWER_PARAMS.items():
                row[name] = data[p]["ANN"]
                # monthly values kept for the seasonal chart on the site report
                row[f"{name}_monthly"] = [data[p][m] for m in
                                          ("JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                                           "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")]
            cache.write_text(json.dumps(row))
            return row
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"NASA POWER request failed for ({lat}, {lon}): {last}")


def build_power_grid(bounds: tuple[float, float, float, float], step: float = config.POWER_GRID_STEP_DEG,
                     workers: int = 4, verbose: bool = True) -> pd.DataFrame:
    """Fetch the POWER lattice covering `bounds` = (min_lon, min_lat, max_lon, max_lat)
    padded by one cell, and save it to data/raw/nasa_power_grid.csv."""
    min_lon, min_lat, max_lon, max_lat = bounds
    lats = np.arange(np.floor(min_lat / step) * step - step, max_lat + 2 * step, step)
    lons = np.arange(np.floor(min_lon / step) * step - step, max_lon + 2 * step, step)
    jobs = [(round(float(la), 2), round(float(lo), 2)) for la in lats for lo in lons]
    if verbose:
        print(f"[power] lattice {len(lats)} x {len(lons)} = {len(jobs)} points")

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_fetch_point, la, lo) for la, lo in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if verbose and n % 25 == 0:
                print(f"[power]   {n}/{len(jobs)}")
    df = pd.DataFrame(rows).sort_values(["lat", "lon"]).reset_index(drop=True)
    flat = df.copy()
    for name in POWER_PARAMS.values():
        flat[f"{name}_monthly"] = flat[f"{name}_monthly"].apply(json.dumps)
    flat.to_csv(config.POWER_GRID_CSV, index=False)
    return df


def load_power_grid() -> pd.DataFrame:
    df = pd.read_csv(config.POWER_GRID_CSV)
    for name in POWER_PARAMS.values():
        df[f"{name}_monthly"] = df[f"{name}_monthly"].apply(json.loads)
    return df


class PowerInterpolator:
    """Bilinear interpolation of the POWER lattice to arbitrary coordinates."""

    def __init__(self, grid: pd.DataFrame | None = None):
        grid = grid if grid is not None else load_power_grid()
        self.lats = np.sort(grid["lat"].unique())
        self.lons = np.sort(grid["lon"].unique())
        self.grid = grid.set_index(["lat", "lon"])
        self._interp = {}
        for name in POWER_PARAMS.values():
            z = grid.pivot(index="lat", columns="lon", values=name).loc[self.lats, self.lons].to_numpy()
            self._interp[name] = RegularGridInterpolator((self.lats, self.lons), z,
                                                         bounds_error=False, fill_value=None)
            mz = np.stack(
                grid.pivot(index="lat", columns="lon", values=f"{name}_monthly")
                .loc[self.lats, self.lons].to_numpy().ravel()
            ).reshape(len(self.lats), len(self.lons), 12)
            self._interp[f"{name}_monthly"] = RegularGridInterpolator((self.lats, self.lons), mz,
                                                                      bounds_error=False, fill_value=None)

    def __call__(self, lat, lon) -> pd.DataFrame:
        pts = np.column_stack([np.atleast_1d(lat), np.atleast_1d(lon)])
        return pd.DataFrame({name: self._interp[name](pts) for name in POWER_PARAMS.values()})

    def monthly(self, lat: float, lon: float) -> dict[str, list[float]]:
        return {name: [round(float(v), 3) for v in self._interp[f"{name}_monthly"]([[lat, lon]])[0]]
                for name in POWER_PARAMS.values()}
