"""
Global Earth Engine feature stack.

Every predictor comes from a dataset with worldwide coverage inside the Earth
Engine catalog, so features can be extracted for *any* coordinate on land in a
single request - no per-point calls to external APIs.

Static (time-invariant within the study horizon)
  Copernicus DEM GLO-30 (2024) elevation, slope, aspect (30 m, covers > 60 N)
  ERA5-Land monthly            GHI (surface_solar_radiation_downwards), 2 m air temp.
  MODIS MOD08_M3               cloud fraction (1 deg)
  MAP accessibility 2015       travel time to nearest city (1 km)
  RESOLVE Ecoregions 2017      biome id (metadata for stratified reporting only)

Epoch-dependent (evaluated either at a pre-construction baseline or "now")
  MODIS MOD13A1 (500 m)        NDVI
  MODIS MOD09A1 (500 m)        NDBI from NIR (b2) and SWIR-1 (b6)
  MODIS MOD11A2 (1 km)         daytime land surface temperature
  MODIS MCD12Q1 (500 m)        IGBP land cover
  VIIRS DNB monthly (~500 m)   night-time lights radiance
  GHSL P2023A (100 m)          population
"""
from __future__ import annotations

import ee

from . import config
from .gee import init_ee, worldcover

ACCESSIBILITY_ASSET = "projects/malariaatlasproject/assets/accessibility/accessibility_to_cities/2015_v1_0"

STATIC_BANDS = ["elevation", "slope", "aspect", "ghi", "temperature", "cloud_cover", "accessibility"]
EPOCH_BANDS = ["ndvi", "ndbi", "lst", "landcover", "nightlights", "population"]
META_BANDS = ["biome", "worldcover"]


# --------------------------------------------------------------------------- #
# Static layers
# --------------------------------------------------------------------------- #
def copernicus_terrain() -> ee.Image:
    """Copernicus GLO-30 DSM mosaic with slope/aspect.

    The collection is a set of 1x1 deg tiles; after mosaicking the image has no
    meaningful default projection, so we restore the tiles' native projection
    before ee.Terrain computes 3x3-kernel slope/aspect in metres.
    """
    col = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").select("DEM")
    dem = col.mosaic().setDefaultProjection(col.first().projection()).rename("elevation")
    return dem.addBands(ee.Terrain.slope(dem).rename("slope")).addBands(ee.Terrain.aspect(dem).rename("aspect"))


def _fill_coast(img: ee.Image, proj: ee.Projection, radius_px: int = 3) -> ee.Image:
    """ERA5-Land is land-only at ~9 km, so thin coastal strips/islands are masked.
    Fill them from a focal mean computed in the native grid."""
    native = img.reproject(proj)
    return native.unmask(native.focalMean(radius_px, "square", "pixels").reproject(proj))


def era5_climate() -> ee.Image:
    """Long-term mean GHI (kWh/m2/day) and 2 m temperature (deg C) from ERA5-Land.

    surface_solar_radiation_downwards_sum is accumulated per month in J/m2:
        GHI [kWh m-2 day-1] = mean_monthly_sum / 3.6e6 / 30.4375
    """
    col = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterDate(*config.CLIMATOLOGY_YEARS)
    proj = col.first().select("temperature_2m").projection()
    # Some monthly images store sea pixels as unmasked zeros; mask them per image
    # so they do not drag the long-term mean down along coastlines.
    ssrd = col.select("surface_solar_radiation_downwards_sum").map(lambda i: i.updateMask(i.gt(0)))
    tk = col.select("temperature_2m").map(lambda i: i.updateMask(i.gt(150)))
    ghi = ssrd.mean().divide(3.6e6).divide(30.4375)
    t2m = tk.mean().subtract(273.15)
    return _fill_coast(ghi, proj).rename("ghi").addBands(_fill_coast(t2m, proj).rename("temperature"))


def modis_cloud() -> ee.Image:
    """Mean cloud fraction (%) from the MODIS/Terra monthly atmosphere product."""
    col = ee.ImageCollection("MODIS/061/MOD08_M3").filterDate(*config.CLIMATOLOGY_YEARS)
    return col.select("Cloud_Fraction_Mean_Mean").mean().multiply(0.0001 * 100).rename("cloud_cover")


def accessibility() -> ee.Image:
    img = ee.Image(ACCESSIBILITY_ASSET).select(0).rename("accessibility")
    return img.updateMask(img.gte(0))


def biome() -> ee.Image:
    eco = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")
    return eco.reduceToImage(["BIOME_NUM"], ee.Reducer.first()).rename("biome")


# --------------------------------------------------------------------------- #
# Epoch-dependent layers
# --------------------------------------------------------------------------- #
def modis_ndvi(start: str, end: str) -> ee.Image:
    col = ee.ImageCollection("MODIS/061/MOD13A1").filterDate(start, end)
    # SummaryQA 0 = good, 1 = marginal; drop snow/cloud (2, 3)
    col = col.map(lambda i: i.updateMask(i.select("SummaryQA").lte(1)))
    return col.select("NDVI").median().multiply(0.0001).rename("ndvi")


def modis_ndbi(start: str, end: str) -> ee.Image:
    """NDBI = (SWIR1 - NIR) / (SWIR1 + NIR) using MOD09A1 b06 (1628-1652 nm)
    and b02 (841-876 nm); cloudy / shadowed pixels removed with StateQA bits 0-2."""
    def mask(i):
        qa = i.select("StateQA")
        clear = qa.bitwiseAnd(3).eq(0).And(qa.bitwiseAnd(1 << 2).eq(0))
        return i.updateMask(clear)
    col = ee.ImageCollection("MODIS/061/MOD09A1").filterDate(start, end).map(mask)
    med = col.select(["sur_refl_b02", "sur_refl_b06"]).median()
    return med.normalizedDifference(["sur_refl_b06", "sur_refl_b02"]).rename("ndbi")


def modis_lst(start: str, end: str) -> ee.Image:
    """Daytime LST (deg C): DN * 0.02 K, pixels with QC bits 0-1 in {0, 1} kept."""
    def mask(i):
        return i.updateMask(i.select("QC_Day").bitwiseAnd(3).lte(1))
    col = ee.ImageCollection("MODIS/061/MOD11A2").filterDate(start, end).map(mask)
    return col.select("LST_Day_1km").median().multiply(0.02).subtract(273.15).rename("lst")


def modis_landcover(year: int) -> ee.Image:
    img = ee.ImageCollection("MODIS/061/MCD12Q1").filterDate(f"{year}-01-01", f"{year + 1}-01-01").first()
    return img.select("LC_Type1").rename("landcover")


def viirs_lights(start: str, end: str) -> ee.Image:
    """Median monthly VIIRS DNB radiance (nW/cm2/sr), stray-light corrected."""
    col = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate(start, end)
    return col.select("avg_rad").median().max(0).unmask(0).rename("nightlights")


def ghsl_population(epoch: str) -> ee.Image:
    """GHSL residential population, converted from people per 100 m cell to people/km2."""
    img = ee.ImageCollection("JRC/GHSL/P2023A/GHS_POP").filter(ee.Filter.eq("system:index", epoch)).first()
    return img.select("population_count").multiply(100).unmask(0).rename("population")


def epoch_layers(epoch: dict) -> ee.Image:
    s, e = epoch["composite"]
    ls, le = epoch["lights"]
    return (
        modis_ndvi(s, e)
        .addBands(modis_ndbi(s, e))
        .addBands(modis_lst(s, e))
        .addBands(modis_landcover(epoch["landcover_year"]))
        .addBands(viirs_lights(ls, le))
        .addBands(ghsl_population(epoch["population_epoch"]))
    )


# --------------------------------------------------------------------------- #
# Stacks
# --------------------------------------------------------------------------- #
def static_layers() -> ee.Image:
    return (
        copernicus_terrain()
        .addBands(era5_climate())
        .addBands(modis_cloud())
        .addBands(accessibility())
    )


def training_image() -> ee.Image:
    """Static + baseline (suffix _pre) + current epoch + metadata bands.
    Used once to build the training table, so both epochs are available for
    the leakage ablation."""
    init_ee()
    pre = epoch_layers(config.GLOBAL_BASELINE).rename([f"{b}_pre" for b in EPOCH_BANDS])
    cur = epoch_layers(config.GLOBAL_CURRENT)
    return (static_layers().addBands(pre).addBands(cur)
            .addBands(biome()).addBands(worldcover().rename("worldcover"))).float()


def prediction_image() -> ee.Image:
    """Static + current epoch (+ WorldCover for display) - used for live
    predictions and the global heatmap."""
    init_ee()
    return (static_layers().addBands(epoch_layers(config.GLOBAL_CURRENT))
            .addBands(biome()).addBands(worldcover().rename("worldcover"))).float()


TRAINING_BANDS = STATIC_BANDS + [f"{b}_pre" for b in EPOCH_BANDS] + EPOCH_BANDS + META_BANDS
PREDICTION_BANDS = STATIC_BANDS + EPOCH_BANDS + META_BANDS
