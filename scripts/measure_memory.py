"""
Measure the API's resident memory at startup, after a prediction and after a
PDF report (the steps that pull in SHAP and matplotlib). Starts its own uvicorn
on a spare port with live Earth Engine off, so the number reflects the app only.

    python scripts/measure_memory.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
PORT = 8765


def rss_mb(pid: int) -> float:
    """Resident set size of the uvicorn worker process (Windows + Linux)."""
    try:
        import psutil
        # include children: on Windows the venv python.exe is a launcher for the real interpreter
        procs = [psutil.Process(pid)] + psutil.Process(pid).children(recursive=True)
        return sum(p.memory_info().rss for p in procs) / 2**20
    except ImportError:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1024
    return float("nan")


def main():
    env = {**os.environ, "LIVE_GEE": "0"}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.app.main:app", "--port", str(PORT)],
                            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{PORT}"
    try:
        for _ in range(120):
            try:
                if httpx.get(base + "/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.5)
        time.sleep(1.0)
        print(f"after startup     : {rss_mb(proc.pid):6.0f} MB")
        r = httpx.post(base + "/predict", json={"lat": 24.411, "lon": 32.7102}, timeout=60)
        r.raise_for_status()
        print(f"after /predict    : {rss_mb(proc.pid):6.0f} MB   (score {r.json()['score']})")
        r = httpx.get(base + "/report/24.411/32.7102", timeout=60)
        r.raise_for_status()
        print(f"after /report PDF : {rss_mb(proc.pid):6.0f} MB   ({len(r.content) // 1024} KB PDF)")
        mods = httpx.get(base + "/health", timeout=5).json()
        print("health:", mods.get("status"))
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
