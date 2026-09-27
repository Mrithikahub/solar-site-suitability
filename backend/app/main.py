"""
Solar Site Suitability API (FastAPI).

Run locally from the repository root:
    uvicorn backend.app.main:app --reload --port 8000

Endpoints
  GET  /health                     model + data status, live-GEE state, cache stats
  POST /predict                    {lat, lon, area_m2?, efficiency?, performance_ratio?}
  GET  /heatmap?scope=global       suitability grid (JSON) + overlay image URL and bounds
  GET  /heatmap/image?scope=...    the heatmap overlay image
  GET  /layers?scope=...           catalogue of map layers (NDVI, slope, GHI, land cover, ...)
  GET  /layers/{name}?scope=...    one layer's metadata (legend, bounds, image URL)
  GET  /layers/{name}/image        the layer overlay image (webp or png)
  POST /compare                    2-5 sites side by side, ranked
  GET  /report/{lat}/{lon}         downloadable PDF site report
  GET  /geocode?q=...              place search (Photon + Nominatim, curated solar parks first)
  GET  /places/famous              curated famous solar parks
  GET  /metrics                    model evaluation + AHP results (methodology page)
  GET  /figures/{file}             report figures
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from fastapi import FastAPI, HTTPException, Query, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

from model import config  # noqa: E402
from model.predict import Predictor  # noqa: E402

from . import geocode  # noqa: E402
from .cache import TTLCache  # noqa: E402
from .report import build_pdf  # noqa: E402
from .schemas import CompareRequest, PredictRequest, Scope  # noqa: E402

log = logging.getLogger("solar-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

LAYERS_DIR = config.PROCESSED_DIR / "layers"
STATE: dict = {}
predict_cache = TTLCache(maxsize=4096, ttl_s=24 * 3600)
report_cache = TTLCache(maxsize=256, ttl_s=24 * 3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    STATE["predictor"] = Predictor()
    STATE["manifest"] = json.loads((LAYERS_DIR / "manifest.json").read_text())
    STATE["heatmap_global"] = json.loads((config.PROCESSED_DIR / "global_heatmap_0p5.json").read_text())
    STATE["metrics"] = json.loads((config.PROCESSED_DIR / "metrics.json").read_text())
    STATE["physical"] = json.loads((config.PROCESSED_DIR / "physical_model.json").read_text())
    STATE["ahp"] = json.loads((config.PROCESSED_DIR / "ahp_results.json").read_text())
    STATE["started"] = time.time()
    log.info("models and data loaded in %.1fs (live GEE %s)", time.time() - t0,
             "enabled" if STATE["predictor"].enable_live else "disabled")
    yield
    STATE.clear()


app = FastAPI(title="Solar Site Suitability API", version="1.0.0",
              description="Remote sensing + machine learning solar PV site suitability for any location on Earth.",
              lifespan=lifespan)

# CORS. Browsers send the Origin without a trailing slash, so configured values are
# normalised ("https://x.vercel.app/" -> "https://x.vercel.app"). Vercel production and
# preview URLs (*.vercel.app) and local dev servers are always allowed, so a missing or
# mistyped ALLOWED_ORIGINS on the host cannot take the site offline. The API is public
# and read-only (no cookies / credentials), so a permissive origin list is safe.
DEFAULT_ORIGIN_REGEX = r"https://([a-z0-9-]+\.)*vercel\.app|http://(localhost|127\.0\.0\.1)(:\d+)?"
CORS_ORIGINS = sorted({o.strip().rstrip("/") for o in
                       os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",") if o.strip()})
_extra_regex = (os.getenv("ALLOWED_ORIGIN_REGEX") or "").strip()
CORS_ORIGIN_REGEX = f"(?:{DEFAULT_ORIGIN_REGEX})" + (f"|(?:{_extra_regex})" if _extra_regex else "")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_origin_regex=CORS_ORIGIN_REGEX,
                   allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"], max_age=3600)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.mount("/figures", StaticFiles(directory=config.FIGURES_DIR), name="figures")


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
@app.exception_handler(ValueError)
async def value_error(_: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"error": "invalid_input", "detail": str(exc)})


@app.exception_handler(LookupError)
async def lookup_error(_: Request, exc: LookupError):
    return JSONResponse(status_code=404, content={"error": "no_data", "detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    msgs = "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors())
    return JSONResponse(status_code=422, content={"error": "invalid_request", "detail": msgs})


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": type(exc).__name__})


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _predict_cached(lat: float, lon: float, area_m2: float, efficiency: float, pr: float) -> dict:
    key = (round(lat, 4), round(lon, 4), round(area_m2, 1), round(efficiency, 4), round(pr, 4))
    res, hit = predict_cache.get_or_set(
        key, lambda: STATE["predictor"].predict(lat, lon, area_m2, efficiency, pr))
    return {**res, "cache_hit": hit}


def _layer_entry(scope: str, name: str, request: Request) -> dict:
    layers = STATE["manifest"].get(scope)
    if layers is None:
        raise HTTPException(404, detail=f"unknown scope '{scope}'")
    if name not in layers:
        raise HTTPException(404, detail=f"unknown layer '{name}' for scope '{scope}'. "
                                        f"Available: {', '.join(layers)}")
    e = dict(layers[name])
    base = str(request.base_url).rstrip("/")
    e["name"] = name
    e["scope"] = scope
    e["image_url"] = f"{base}/layers/{name}/image?scope={scope}"
    e.pop("vis", None)
    return e


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    p: Predictor = STATE["predictor"]
    return {
        "status": "ok", "uptime_s": round(time.time() - STATE["started"]),
        "model": {"suitability": p.physical["name"], "development": p.model_name,
                  "tamil_nadu": p.tn["bundle"]["model_name"] if p.tn else None},
        "grid_cells": int(len(p.grid)),
        "live_gee": {"enabled": p.enable_live, "timeout_s": p.live_timeout_s,
                     "paused_for_s": max(0, round(p._gee_disabled_until - time.time())),
                     "last_error": p.last_live_error},
        "cache": {"predict": predict_cache.stats(), "report": report_cache.stats()},
        "cors": {"origins": CORS_ORIGINS, "origin_regex": CORS_ORIGIN_REGEX},
    }


@app.post("/predict")
async def predict(req: PredictRequest):
    return await run_in_threadpool(_predict_cached, req.lat, req.lon, req.area_m2, req.efficiency,
                                   req.performance_ratio)


@app.get("/heatmap")
def heatmap(request: Request, scope: Scope = "global",
            metric: str = Query("suitability", pattern="^(suitability|development|ahp)$"),
            include_grid: bool = True):
    """Suitability grid. The global grid carries all three scores per 0.5° cell
    (suitability, development, ahp) as parallel arrays with lat / lon."""
    if scope == "tamil_nadu" and metric == "development":
        raise HTTPException(404, detail="development likelihood is only available for the global scope")
    entry = _layer_entry(scope, metric, request)
    out = {"scope": scope, "metric": metric, "layer": entry,
           "legend": {"min": 0, "max": 100, "palette": entry["palette"], "classes": STATE["predictor"].thresholds}}
    if scope == "global" and include_grid:
        out["grid"] = STATE["heatmap_global"]
    return out


@app.get("/heatmap/image")
def heatmap_image(scope: Scope = "global",
                  metric: str = Query("suitability", pattern="^(suitability|development|ahp)$"),
                  fmt: str = Query("webp", pattern="^(webp|png)$")):
    return layer_image(metric, scope, fmt)


@app.get("/layers")
def layers(request: Request, scope: Scope = "global"):
    return {"scope": scope, "layers": [_layer_entry(scope, n, request) for n in STATE["manifest"][scope]]}


@app.get("/layers/{name}")
def layer(name: str, request: Request, scope: Scope = "global"):
    return _layer_entry(scope, name, request)


@app.get("/layers/{name}/image")
def layer_image(name: str, scope: Scope = "global", fmt: str = Query("webp", pattern="^(webp|png)$")):
    layers_ = STATE["manifest"].get(scope, {})
    if name not in layers_:
        raise HTTPException(404, detail=f"unknown layer '{name}' for scope '{scope}'")
    path = LAYERS_DIR / Path(layers_[name]["file"]).with_suffix(f".{fmt}").name
    if not path.exists():
        path = LAYERS_DIR / layers_[name]["file"]
    return FileResponse(path, media_type=f"image/{path.suffix.lstrip('.')}",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.post("/compare")
async def compare(req: CompareRequest):
    def run():
        results = []
        for i, s in enumerate(req.sites):
            try:
                r = _predict_cached(s.lat, s.lon, req.area_m2, req.efficiency, req.performance_ratio)
                results.append({"index": i, "name": s.name or f"Site {chr(65 + i)}", "ok": True, "result": r})
            except (ValueError, LookupError) as e:
                results.append({"index": i, "name": s.name or f"Site {chr(65 + i)}", "ok": False, "error": str(e)})
        ok = [r for r in results if r["ok"]]
        ranking = sorted(ok, key=lambda r: -r["result"]["score"])
        for rank, r in enumerate(ranking, 1):
            r["rank"] = rank
        best_energy = max(ok, key=lambda r: (r["result"]["energy"] or {}).get("annual_kwh", 0), default=None)
        return {"sites": results,
                "summary": {"best_score": ranking[0]["name"] if ranking else None,
                            "best_energy": best_energy["name"] if best_energy else None,
                            "n_ok": len(ok)}}
    return await run_in_threadpool(run)


@app.get("/report/{lat}/{lon}")
async def report(lat: float, lon: float, area_m2: float = Query(4046.8564224, gt=0, le=1e9),
                 efficiency: float = Query(0.20, gt=0, le=0.5), performance_ratio: float = Query(0.75, gt=0, le=1),
                 name: str | None = Query(None, max_length=120)):
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("latitude must be in [-90, 90] and longitude in [-180, 180]")

    def run():
        key = (round(lat, 4), round(lon, 4), round(area_m2, 1), round(efficiency, 4), round(performance_ratio, 4),
               name or "")
        pdf, _ = report_cache.get_or_set(
            key, lambda: build_pdf(_predict_cached(lat, lon, area_m2, efficiency, performance_ratio), name))
        return pdf
    pdf = await run_in_threadpool(run)
    fname = f"solar_site_report_{lat:.4f}_{lon:.4f}.pdf"
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@app.get("/geocode")
async def geocode_search(q: str = Query(..., min_length=2, max_length=200), limit: int = Query(5, ge=1, le=10)):
    try:
        return {"query": q, "results": await run_in_threadpool(geocode.search, q, limit)}
    except Exception as e:
        raise HTTPException(502, detail=f"geocoding service unavailable: {type(e).__name__}")


@app.get("/places/famous")
def places_famous():
    """Curated famous solar parks (quick-pick chips on the map)."""
    return {"places": geocode.famous_parks()}


@app.get("/metrics")
def metrics():
    return {"ml": STATE["metrics"], "ahp": STATE["ahp"], "physical": STATE["physical"]}


@app.get("/")
def root():
    return {"name": "Solar Site Suitability API", "docs": "/docs", "health": "/health"}
