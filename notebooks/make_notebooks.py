"""
Generates the project notebooks from plain Python cell lists so they stay in
sync with the `model` package. Run, then execute with nbconvert:

    python notebooks/make_notebooks.py 01
    jupyter nbconvert --to notebook --execute --inplace notebooks/01_gee_extraction.ipynb
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

SETUP = """\
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from model import config
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 200, 'font.family': 'DejaVu Sans',
                     'axes.spines.top': False, 'axes.spines.right': False})
"""


def nb_01() -> nbf.NotebookNode:
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md("""# 01 · Remote-sensing data extraction (Google Earth Engine)

**Goal:** build a labelled point dataset for Tamil Nadu in which every point carries the physical,
spectral, climatic and infrastructure characteristics that decide whether land is suitable for a
utility-scale solar PV plant.

| Group | Predictor | Sensor / product | Native resolution | Why it matters |
|---|---|---|---|---|
| Climate | GHI, air temperature, cloud cover | NASA POWER (CERES / MERRA-2) | 1° / 0.5° | energy yield, panel efficiency loss with heat |
| Terrain | elevation, slope, aspect | SRTM GL1 v3 (C-band InSAR) | 30 m | construction cost, self-shading |
| Spectral | NDVI, NDBI | Sentinel-2 MSI L2A | 10–20 m | vegetation / built-up land to avoid |
| Thermal | land surface temperature | Landsat 8/9 TIRS C2 L2 | 100 m (resampled 30 m) | surface heat regime |
| Land cover | 11-class map | ESA WorldCover v200 (2021) | 10 m | legal / practical exclusions |
| Infrastructure | distance to road, power line, substation | OpenStreetMap | vector | access & grid connection |

**Labels:** positives = points inside OSM-mapped solar plants (`power=plant` + `plant:source=solar`
and ground-mounted `generator:source=solar` arrays); negatives = random points across the state,
≥ 1 km from any plant."""),
        code(SETUP),
        md("## 1. Earth Engine & study region"),
        code("""from model import gee, osm
gee.init_ee()
gj = gee.save_region_geojson()
region = osm.region_polygon_from_geojson(gj)
region_gdf = gpd.GeoDataFrame(geometry=[region], crs=4326)
area_km2 = region_gdf.to_crs(config.METRIC_EPSG).area.iloc[0] / 1e6
print(f"{config.REGION_NAME}: {area_km2:,.0f} km²  |  bounds {tuple(round(b, 2) for b in region.bounds)}")"""),
        md("""## 2. Earth Engine feature stack

All raster predictors are combined into **one multi-band `ee.Image`**, so a single `reduceRegions`
call returns every band for hundreds of points at once.

* **Sentinel-2** — `COPERNICUS/S2_SR_HARMONIZED`, Sep 2025 → Sep 2026, scenes with < 60 % cloud,
  per-pixel mask from **Cloud Score+** (`cs_cdf ≥ 0.60`), reflectance = DN / 10 000, median composite.
  NDVI = (B8 − B4)/(B8 + B4); NDBI = (B11 − B8)/(B11 + B8). B11 (20 m) is resampled to 10 m.
* **Landsat 8/9** — Collection-2 Level-2; clouds, cirrus, shadows and fill removed with `QA_PIXEL`
  bits 0–4; LST (°C) = `ST_B10` × 0.00341802 + 149.0 − 273.15; 12-month median.
* **SRTM** — slope and aspect from `ee.Terrain` (3 × 3 finite-difference kernel).
* **WorldCover** — `ESA/WorldCover/v200`, band `Map`.
* **Baseline (ablation)** — Landsat 8 2013-04 → 2015-03 NDVI, NDBI, LST from *before* most plants
  were built, to test whether the model is learning "suitability" or just "looks like panels"."""),
        code("""stack = gee.feature_image()
print('bands:', stack.bandNames().getInfo())
s2_count = (gee.ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(gee.region_geometry())
            .filterDate(config.COMPOSITE_START, config.COMPOSITE_END)
            .filter(gee.ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)).size().getInfo())
ls_count = (gee.ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').merge(gee.ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
            .filterBounds(gee.region_geometry())
            .filterDate(config.COMPOSITE_START, config.COMPOSITE_END)
            .filter(gee.ee.Filter.lt('CLOUD_COVER', 70)).size().getInfo())
print(f'Sentinel-2 scenes in composite: {s2_count}  |  Landsat 8/9 scenes: {ls_count}')"""),
        md("### Layer previews (rendered by Earth Engine, stitched from Mercator tiles)"),
        code("""from PIL import Image
layers = json.loads((config.PROCESSED_DIR / 'layers' / 'layers.json').read_text())
titles = {'truecolor': 'Sentinel-2 true colour (B4-B3-B2)', 'ndvi': 'NDVI (Sentinel-2)',
          'ndbi': 'NDBI (Sentinel-2)', 'lst': 'LST °C (Landsat 8/9)', 'slope': 'Slope ° (SRTM)',
          'elevation': 'Elevation m (SRTM)', 'landcover': 'ESA WorldCover 2021'}
order = ['truecolor', 'ndvi', 'ndbi', 'lst', 'slope', 'landcover']
fig, axes = plt.subplots(2, 3, figsize=(15, 12))
for ax, name in zip(axes.ravel(), order):
    ax.imshow(Image.open(config.PROCESSED_DIR / 'layers' / layers[name]['file']))
    ax.set_title(titles[name], fontsize=12, loc='left'); ax.axis('off')
plt.tight_layout()
plt.savefig(config.FIGURES_DIR / 'rs_layers_overview.png', bbox_inches='tight')
plt.show()"""),
        md("## 3. Training labels from OpenStreetMap"),
        code("""sites = gpd.read_file(config.SOLAR_POLYGONS_GEOJSON)
df = pd.read_csv(config.DATASET_CSV)
print(f"solar sites: {len(sites)}   total footprint: {sites['area_m2'].sum()/1e6:.1f} km²")
print(f"largest sites:")
display(sites.sort_values('area_m2', ascending=False)[['site_id', 'name', 'area_m2', 'start_date']]
        .head(8).assign(area_km2=lambda d: (d.area_m2 / 1e6).round(2)).drop(columns='area_m2'))
print(df['label'].value_counts().rename({1: 'solar (positive)', 0: 'non-solar (negative)'}))"""),
        code("""fig, ax = plt.subplots(figsize=(8, 10))
region_gdf.boundary.plot(ax=ax, color='#334155', lw=0.8)
neg, pos = df[df.label == 0], df[df.label == 1]
ax.scatter(neg.lon, neg.lat, s=3, c='#94a3b8', alpha=0.6, label=f'negative (n={len(neg)})')
ax.scatter(pos.lon, pos.lat, s=6, c='#f59e0b', alpha=0.9, label=f'positive (n={len(pos)})')
ax.set_title(f'Training samples - {config.REGION_NAME}', loc='left')
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); ax.legend(loc='lower left', markerscale=3)
ax.set_aspect(1 / np.cos(np.deg2rad(11)))
plt.savefig(config.FIGURES_DIR / 'training_samples_map.png', bbox_inches='tight'); plt.show()"""),
        md("## 4. Final dataset"),
        code("""from model.features import RAW_FEATURE_COLS
print(df.shape)
display(df.head())
display(df[RAW_FEATURE_COLS].describe().T.round(3))
print('missing values per column:')
print(df[RAW_FEATURE_COLS].isna().sum()[lambda s: s > 0])"""),
        code("""summary = df.groupby('label')[RAW_FEATURE_COLS].median().T
summary.columns = ['non-solar median', 'solar median']
summary.round(3)"""),

        # ------------------------------------------------------------------ GLOBAL
        md("""---
# Part B · Global dataset

To make the model work **anywhere on Earth**, every predictor is swapped for a dataset with global
coverage inside the Earth Engine catalog, so a single `reduceRegions` request returns all features for
any coordinate — no per-point calls to external APIs.

| Group | Predictor | Global dataset | Resolution |
|---|---|---|---|
| Climate | GHI, 2 m air temperature | ERA5-Land monthly (2005-2024 mean) | ~9 km |
| Climate | cloud fraction | MODIS Terra MOD08_M3 | 1° |
| Terrain | elevation, slope, aspect | Copernicus DEM GLO-30 (X-band TanDEM-X) | 30 m, pole-to-pole |
| Spectral | NDVI | MODIS MOD13A1 | 500 m |
| Spectral | NDBI (b6 SWIR-1, b2 NIR) | MODIS MOD09A1 | 500 m |
| Thermal | daytime LST | MODIS MOD11A2 | 1 km |
| Land cover | IGBP class | MODIS MCD12Q1 (annual) | 500 m |
| Human | travel time to city | Malaria Atlas Project accessibility 2015 | 1 km |
| Human | night-time lights | VIIRS DNB monthly (stray-light corrected) | ~460 m |
| Human | population density | GHSL P2023A | 100 m |

**Labels:** Kruitwagen *et al.* (2021, *Nature*) global PV inventory — 68 661 plant polygons detected
with ML on Sentinel-2 and SPOT imagery. One interior point per plant ≥ 1 ha, stratified by world region.
Negatives: 50 % area-uniform background on land, 50 % *hard* negatives 5–100 km from a plant.

### Label leakage and the pre-construction epoch
Satellites observing an existing plant see **the panels**, not the land that was chosen. If NDVI,
land cover, etc. are measured *now*, a model learns "what a solar farm looks like". Time-varying
predictors are therefore sampled from a **2008–2010 baseline** (before almost all inventory plants were
built); at prediction time the same variables are measured *now* — the land a new plant would replace."""),
        code("""from model.features import IGBP_NAMES
g = pd.read_csv(config.GLOBAL_DATASET_CSV)
print(g.shape)
display(pd.crosstab(g['region'], g['label'].map({1: 'solar', 0: 'non-solar'}), margins=True))
display(g.groupby(['label', 'neg_type'], dropna=False).size().rename('n'))"""),
        code("""regions = gpd.read_file(config.WORLD_REGIONS_GEOJSON)
fig, ax = plt.subplots(figsize=(16, 8))
regions.plot(ax=ax, color='#e2e8f0', edgecolor='#cbd5e1', lw=0.3)
for (lab, nt), c, s, name in [((0, 'background'), '#64748b', 2, 'background negative'),
                              ((0, 'hard'), '#0ea5e9', 2, 'hard negative (5-100 km)'),
                              ((1, None), '#f59e0b', 4, 'solar plant')]:
    sub = g[(g.label == lab) & ((g.neg_type == nt) if nt else True)]
    ax.scatter(sub.lon, sub.lat, s=s, c=c, alpha=0.7, label=f'{name} (n={len(sub):,})')
ax.set_xlim(-180, 180); ax.set_ylim(-58, 75); ax.set_aspect('equal')
ax.legend(loc='lower left', markerscale=4, frameon=False)
ax.set_title('Global training samples', loc='left'); ax.set_xticks([]); ax.set_yticks([])
plt.savefig(config.FIGURES_DIR / 'global_training_samples.png', bbox_inches='tight'); plt.show()"""),
        md("### Evidence of leakage: land cover at plant locations, then vs now"),
        code("""from model.config import WORLDCOVER_CLASSES
pos = g[g.label == 1]
pre = pos['landcover_pre'].round().map(IGBP_NAMES).value_counts(normalize=True).head(8) * 100
now = pos['worldcover'].round().map(WORLDCOVER_CLASSES).value_counts(normalize=True).head(8) * 100
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
axes[0].barh(pre.index[::-1], pre.values[::-1], color='#16a34a')
axes[0].set_title('Before construction - MODIS IGBP 2008 (%)', loc='left')
axes[1].barh(now.index[::-1], now.values[::-1], color='#dc2626')
axes[1].set_title('After construction - ESA WorldCover 2021 (%)', loc='left')
plt.tight_layout(); plt.savefig(config.FIGURES_DIR / 'leakage_landcover_then_vs_now.png', bbox_inches='tight'); plt.show()
cols = ['ndvi', 'ndbi', 'lst', 'nightlights', 'population']
d = pd.DataFrame({'2008-10 (pre)': [pos[f'{c}_pre'].median() for c in cols],
                  'now': [pos[c].median() for c in cols]}, index=cols)
print('Median at plant locations:'); display(d.round(3))"""),
        code("""print('missing values (%):')
print((g.drop(columns=['install_date', 'lc_before', 'neg_type', 'unique_id', 'area_m2', 'capacity_mw'])
       .isna().mean() * 100).round(2)[lambda s: s > 0])"""),
    ]
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "solar-venv", "display_name": "Python (solar .venv)", "language": "python"}
    return nb


def _nb(cells) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "solar-venv", "display_name": "Python (solar .venv)", "language": "python"}
    return nb


def nb_02() -> nbf.NotebookNode:
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    return _nb([
        md("""# 02 · Exploratory data analysis

How do solar-plant locations differ from non-solar land in each predictor? We compare class-conditional
distributions, check correlations (multicollinearity matters for interpreting importances), and look at how
the time-varying remote-sensing indices changed between the pre-construction baseline and today."""),
        code(SETUP + "\nfrom model import viz\nviz.apply_style()\nfrom model.features import RAW_FEATURE_COLS, FEATURE_META, GLOBAL_FEATURE_META"),
        md("## Tamil Nadu case study"),
        code("""t = pd.read_csv(config.DATASET_CSV)
cols = ['ghi', 'temperature', 'slope', 'elevation', 'ndvi_pre', 'ndbi_pre', 'lst_pre',
        'dist_road_km', 'dist_powerline_km', 'dist_substation_km']
fig, axes = plt.subplots(2, 5, figsize=(17, 6.4))
for ax, c in zip(axes.ravel(), cols):
    lo, hi = t[c].quantile([0.01, 0.99])
    bins = np.linspace(lo, hi, 40)
    for lab, color, name in ((0, viz.SERIES[0], 'non-solar'), (1, viz.SERIES[1], 'solar')):
        ax.hist(t.loc[t.label == lab, c].clip(lo, hi), bins=bins, density=True, alpha=0.55, color=color, label=name)
    base = c.replace('_pre', '')
    ax.set_title(FEATURE_META.get(c, FEATURE_META.get(base, {'label': c}))['label'], fontsize=10)
    ax.set_yticks([])
axes[0, 0].legend()
plt.suptitle('Tamil Nadu - class-conditional distributions (spectral = 2013-15 Landsat baseline)', x=0.01, ha='left',
             fontweight='bold')
plt.tight_layout(); plt.savefig(config.FIGURES_DIR / 'eda_distributions_tamil_nadu.png', bbox_inches='tight'); plt.show()"""),
        code("""num = t[['ghi', 'temperature', 'cloud_cover', 'elevation', 'slope', 'ndvi_pre', 'ndbi_pre', 'lst_pre',
         'dist_road_km', 'dist_powerline_km', 'dist_substation_km']]
corr = num.corr(method='spearman')
fig, ax = plt.subplots(figsize=(8.5, 7)); ax.grid(False)
im = ax.imshow(corr, cmap=viz.SHAP_CMAP, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha='right'); ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', fontsize=7.5)
plt.colorbar(im, fraction=0.04); ax.set_title('Spearman correlation - Tamil Nadu predictors')
plt.tight_layout(); plt.savefig(config.FIGURES_DIR / 'eda_correlation_tamil_nadu.png', bbox_inches='tight'); plt.show()"""),
        md("""### Leakage in 10 m Sentinel-2 / 30 m Landsat data
At solar sites the *current* NDBI and LST jump because the sensor sees panels and gravel, while negatives barely
change. This is why the Tamil Nadu model is trained on the 2013-15 values."""),
        code("""rows = []
for c in ['ndvi', 'ndbi', 'lst']:
    for lab in (0, 1):
        sub = t[t.label == lab]
        rows.append({'index': c.upper(), 'class': 'solar' if lab else 'non-solar',
                     '2013-15 median': sub[f'{c}_pre'].median(), 'now median': sub[c].median(),
                     'change': sub[c].median() - sub[f'{c}_pre'].median()})
pd.DataFrame(rows).round(3)"""),
        md("## Global dataset"),
        code("""g = pd.read_csv(config.GLOBAL_DATASET_CSV)
g['log_access'] = np.log1p(g['accessibility'].clip(lower=0))
g['log_lights_pre'] = np.log1p(g['nightlights_pre'].clip(lower=0))
g['log_pop_pre'] = np.log1p(g['population_pre'].clip(lower=0))
cols = ['ghi', 'temperature', 'cloud_cover', 'slope', 'ndvi_pre', 'lst_pre', 'log_access', 'log_lights_pre', 'log_pop_pre', 'elevation']
titles = ['GHI (kWh/m²/day)', 'Air temperature (°C)', 'Cloud fraction (%)', 'Slope (°)', 'NDVI 2008-09', 'LST 2008-09 (°C)',
          'log travel time to city', 'log night lights 2014', 'log population 2010', 'Elevation (m)']
fig, axes = plt.subplots(2, 5, figsize=(17, 6.4))
for ax, c, ttl in zip(axes.ravel(), cols, titles):
    lo, hi = g[c].quantile([0.01, 0.99]); bins = np.linspace(lo, hi, 40)
    for (lab, nt), color, name in (((0, 'background'), viz.SERIES[0], 'background neg.'),
                                   ((0, 'hard'), viz.SERIES[2], 'hard neg.'), ((1, None), viz.SERIES[1], 'solar')):
        sub = g[(g.label == lab) & ((g.neg_type == nt) if nt else True)]
        ax.hist(sub[c].clip(lo, hi), bins=bins, density=True, alpha=0.5, color=color, label=name)
    ax.set_title(ttl, fontsize=10); ax.set_yticks([])
axes[0, 0].legend(fontsize=8)
plt.suptitle('Global - class-conditional distributions (time-varying = pre-construction epoch)', x=0.01, ha='left', fontweight='bold')
plt.tight_layout(); plt.savefig(config.FIGURES_DIR / 'eda_distributions_global.png', bbox_inches='tight'); plt.show()"""),
        md("""**Reading the plots.** Background negatives are dominated by remote, cold, forested or desert land (long
travel times, no lights). Hard negatives look much more like plant sites - similar climate and access - which is
what forces the model to learn *local* differences (slope, vegetation, land surface temperature)."""),
        code("""summary = g.groupby('label')[['ghi', 'temperature', 'cloud_cover', 'slope', 'accessibility', 'ndvi_pre',
                              'nightlights_pre', 'population_pre']].median().T
summary.columns = ['non-solar median', 'solar median']; summary.round(3)"""),
    ])


def nb_03() -> nbf.NotebookNode:
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    return _nb([
        md("""# 03 · AHP multi-criteria baseline

The Analytic Hierarchy Process (Saaty, 1980) is the classic GIS approach to site suitability:

1. experts compare criteria pairwise on a 1-9 scale → matrix **A** with $a_{ji} = 1/a_{ij}$
2. weights **w** = principal eigenvector of **A** (normalised)
3. consistency: $CI = (\\lambda_{max} - n)/(n-1)$, $CR = CI/RI$; accept if $CR < 0.10$
4. each criterion is rescaled to [0, 1] by a fuzzy membership function
5. suitability $= 100 \\cdot \\sum_i w_i s_i \\times$ constraint mask (water, wetland, built-up, snow, slope > 15°)

All inputs come from cached data (decoded Earth Engine rasters for Tamil Nadu, the 0.5° grid globally)."""),
        code(SETUP + "\nfrom model import ahp, viz\nfrom IPython.display import Image, display\nviz.apply_style()"),
        code("""res = ahp.main()
for key in ('tamil_nadu', 'global'):
    r = res[key]
    print(f"{key}: lambda_max = {r['lambda_max']:.3f}, CI = {r['CI']:.4f}, RI = {r['RI']}, CR = {r['CR']:.4f}")"""),
        md("## Tamil Nadu"),
        code("""r = res['tamil_nadu']
display(pd.DataFrame(r['matrix'], index=r['criteria'], columns=r['criteria']).round(3))
display(pd.DataFrame({'weight': r['weights'], 'standardisation': r['standardisation']}))"""),
        code("""display(Image(str(config.FIGURES_DIR / 'ahp_matrix_tamil_nadu.png'), width=620))
display(Image(str(config.FIGURES_DIR / 'ahp_weights_tamil_nadu.png'), width=560))
display(Image(str(config.FIGURES_DIR / 'ahp_map_tamil_nadu.png'), width=560))"""),
        md("## Global"),
        code("""r = res['global']
display(pd.DataFrame(r['matrix'], index=r['criteria'], columns=r['criteria']).round(3))
display(pd.DataFrame({'weight': r['weights'], 'standardisation': r['standardisation']}))
display(Image(str(config.FIGURES_DIR / 'ahp_map_global.png'), width=1000))"""),
        md("## How well does AHP rank real solar plants?"),
        code("""from sklearn.metrics import roc_auc_score
for name, path in (('Tamil Nadu', 'tn_ahp_points.csv'), ('Global', 'global_ahp_points.csv')):
    d = pd.read_csv(config.PROCESSED_DIR / path)
    print(f"{name}: AUC of AHP score vs plant labels = {roc_auc_score(d.label, d.ahp_score.fillna(0)):.3f}  "
          f"(mean score plants {d[d.label == 1].ahp_score.mean():.1f} vs non-plants {d[d.label == 0].ahp_score.mean():.1f})")"""),
        md("""AHP encodes *physical* potential with expert weights. It ranks Tamil Nadu plants reasonably well but is
weak globally: it rewards the Sahara, Arabia and central Australia, where almost no utility PV has been built
because there is no grid or demand nearby. Phase 3 shows how data-driven models close this gap."""),
    ])


def nb_04() -> nbf.NotebookNode:
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    return _nb([
        md("""# 04 · Model training, evaluation and explanation

Training itself is scripted in `model/train.py` (run `python -m model.train`, ~5 min). This notebook loads its
outputs (`data/processed/metrics.json`, `model/*.pkl`) and walks through the results:

* Random Forest vs XGBoost, stratified split + stratified 5-fold CV + spatially-blocked 5-fold CV
* label-leakage experiment (clean / leaky / as-specified) and physical-only ablation
* per-region performance with low-sample flags
* ML vs AHP, global vs local model on Tamil Nadu
* SHAP global summaries and a per-site explanation, energy estimate"""),
        code(SETUP + "\nfrom IPython.display import Image, display\nm = json.loads((config.PROCESSED_DIR / 'metrics.json').read_text())"),
        md("## Model comparison"),
        code("""def table(res):
    rows = []
    for v, vd in res['variants'].items():
        for mn, r in vd['models'].items():
            t = r['test']
            rows.append({'variant': v, 'model': mn, 'accuracy': t['accuracy'], 'precision': t['precision'],
                         'recall': t['recall'], 'f1': t['f1'], 'roc_auc': t['roc_auc'],
                         'cv_auc': r['cv_stratified_auc']['mean'], 'spatial_cv_auc': r['cv_spatial_auc']['mean'],
                         'auc_on_preconstruction_land': r.get('test_on_preconstruction_land', {}).get('roc_auc')})
    a = res['ahp_test']
    rows.append({'variant': 'baseline', 'model': 'AHP', 'accuracy': a['accuracy'], 'precision': a['precision'],
                 'recall': a['recall'], 'f1': a['f1'], 'roc_auc': a['roc_auc']})
    return pd.DataFrame(rows).round(3)
print('GLOBAL  (deployed:', m['global']['deployed_model'], ')'); display(table(m['global']))
print('TAMIL NADU  (deployed:', m['tamil_nadu']['deployed_model'], ')'); display(table(m['tamil_nadu']))"""),
        code("""for f in ['roc_global.png', 'roc_tamil_nadu.png', 'ahp_vs_ml_global.png', 'ahp_vs_ml_tamil_nadu.png',
          'confusion_global.png', 'confusion_tamil_nadu.png']:
    display(Image(str(config.FIGURES_DIR / f), width=620))"""),
        md("""## Label leakage
Models trained on *current* imagery score higher on their own test set, but that score is inflated: the imagery
shows the panels. Evaluated on **pre-construction land** - what a prospective site actually looks like - the
leaky models drop below the clean ones."""),
        code("""display(Image(str(config.FIGURES_DIR / 'leakage_experiment.png'), width=900))"""),
        md("## Performance by world region"),
        code("""pr = pd.DataFrame(m['global']['per_region']).sort_values('auc', ascending=False)
display(pr[['region', 'n', 'n_pos', 'auc', 'f1', 'precision', 'recall', 'low_sample', 'note']].round(3))
display(Image(str(config.FIGURES_DIR / 'per_region_auc.png'), width=760))"""),
        md("## Global model vs local model on Tamil Nadu"),
        code("""gl = m['tamil_nadu']['global_vs_local']
display(pd.DataFrame({k: {x: gl[k][x] for x in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')}
                      for k in ('local', 'global', 'ahp')}).T.round(3))
print(gl['note'])"""),
        md("## Explainability"),
        code("""for f in ['feature_importance_global.png', 'shap_summary_global.png',
          'feature_importance_tamil_nadu.png', 'shap_summary_tamil_nadu.png']:
    display(Image(str(config.FIGURES_DIR / f), width=720))"""),
        md("## Per-site prediction, SHAP breakdown and energy estimate"),
        code("""from model.predict import Predictor
p = Predictor(enable_live=False)          # cached grid features - no Earth Engine calls
r = p.predict(9.35, 78.38)                # Kamuthi solar park, Tamil Nadu
print(f"score {r['score']} ({r['class']})  source: {r['provenance']['label']}")
print({k: v['score'] for k, v in r['scores'].items()}, '| TN local:', r.get('tamil_nadu_local', {}).get('score'))
e = r['energy']
print(f"energy: {e['annual_mwh']:.0f} MWh/yr per acre  = {e['area_m2']:.0f} m2 x {e['ghi_kwh_m2_day']:.2f} x 365 x 0.20 x 0.75")
pd.DataFrame(r['shap']['contributions'])[['label', 'value', 'contribution_pct']].head(10).round(3)"""),
        md("## Global heatmap"),
        code("""display(Image(str(config.FIGURES_DIR / 'ml_map_global.png'), width=1000))
display(Image(str(config.FIGURES_DIR / 'ml_map_tamil_nadu.png'), width=520))"""),
    ])


BUILDERS = {"01": ("01_gee_extraction.ipynb", nb_01), "02": ("02_eda.ipynb", nb_02),
            "03": ("03_ahp_baseline.ipynb", nb_03), "04": ("04_training.ipynb", nb_04)}

if __name__ == "__main__":
    keys = sys.argv[1:] or list(BUILDERS)
    for k in keys:
        fname, fn = BUILDERS[k]
        nbf.write(fn(), HERE / fname)
        print("wrote", fname)
