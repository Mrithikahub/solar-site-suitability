/* Typed client for the Solar Site Suitability API (FastAPI backend). */

export const API_URL: string = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '') ?? 'http://localhost:8000'

export type ScoreClass = 'High' | 'Medium' | 'Low'
export type Scope = 'global' | 'tamil_nadu'
export type Metric = 'suitability' | 'development' | 'ahp'

export interface ScoreEntry {
  score: number
  class: ScoreClass
  label: string
  description: string
}

export interface ShapItem {
  feature: string
  label: string
  unit: string
  source: string
  value: number | null
  shap: number
  contribution_pct: number
}

export interface ShapExplanation {
  base_score: number
  score: number
  units: string
  contributions: ShapItem[]
}

export interface Energy {
  area_m2: number
  area_acres: number
  ghi_kwh_m2_day: number
  efficiency: number
  performance_ratio: number
  annual_kwh: number
  annual_mwh: number
  peak_capacity_kwp: number
  specific_yield_kwh_per_kwp: number
  co2_avoided_tonnes: number
  households_powered: number
  formula: string
}

export interface Provenance {
  source: 'live_gee' | 'grid_estimate'
  label: string
  cached: boolean
  latency_s?: number
  cell_lat?: number
  cell_lon?: number
  cell_distance_km?: number
  country?: string | null
  region?: string | null
  live_error?: string | null
}

export interface FeatureMeta {
  label?: string
  unit?: string
  source?: string
}

export interface Prediction {
  lat: number
  lon: number
  score: number
  class: ScoreClass
  excluded: string | null
  /** plain-language explanation when excluded, e.g. "Excluded: urban area, not suitable for ground-mounted solar." */
  exclusion_message?: string | null
  model_score: number
  model: { name: string; variant: string; thresholds: { High: number; Medium: number } }
  features: Record<string, number | string | null>
  feature_meta: Record<string, FeatureMeta>
  shap: ShapExplanation
  development_shap: ShapExplanation
  energy: Energy | null
  provenance: Provenance
  scores: { suitability: ScoreEntry; development: ScoreEntry; ahp?: ScoreEntry }
  /** high-res regional model, only returned inside regions that have 10-30 m layers */
  tamil_nadu_local?: { score: number; class: ScoreClass; model: string; note: string; label?: string }
  cache_hit?: boolean
}

export interface LayerEntry {
  name: string
  scope: Scope
  title: string
  description?: string
  unit: string
  source: string
  file: string
  bounds: { south: number; west: number; north: number; east: number }
  kind: 'continuous' | 'categorical' | 'image'
  min?: number
  max?: number
  palette?: string[]
  classes?: { code: number; name: string; color: string }[]
  image_url: string
}

export interface GeocodeResult {
  name: string
  display_name: string
  lat: number
  lon: number
  /** human-readable OSM type: "stadium", "city", "country", "coordinates" ... */
  type: string
  category?: string
  /** point = POI / building / address (fly in + analyse); area = city / region / country (fit bounds) */
  kind: 'point' | 'area'
  /** [south, north, west, east] */
  bbox: [number, number, number, number] | null
  zoom?: number
  source?: 'photon' | 'nominatim' | 'coordinates' | 'curated'
  capacity_mw?: number
}

/** "51.5294, -0.1727" / "51.53 N 0.17 W" -> a point result (no network needed). */
export function parseCoordinates(q: string): GeocodeResult | null {
  const m = q.trim().match(/^(-?\d{1,3}(?:\.\d+)?)\s*°?\s*([NSns])?\s*[,;\s]\s*(-?\d{1,3}(?:\.\d+)?)\s*°?\s*([EWew])?$/)
  if (!m) return null
  let lat = Number(m[1])
  let lon = Number(m[3])
  if (m[2]?.toUpperCase() === 'S') lat = -Math.abs(lat)
  if (m[4]?.toUpperCase() === 'W') lon = -Math.abs(lon)
  if (Number.isNaN(lat) || Number.isNaN(lon) || Math.abs(lat) > 90 || Math.abs(lon) > 180) return null
  return {
    name: `${lat.toFixed(5)}, ${lon.toFixed(5)}`, display_name: 'Go to these coordinates and analyse', lat, lon,
    type: 'coordinates', kind: 'point', bbox: null, zoom: 16, source: 'coordinates',
  }
}

export interface CompareSiteResult {
  index: number
  name: string
  ok: boolean
  rank?: number
  error?: string
  result?: Prediction
}

export interface EnergyParams {
  area_m2?: number
  efficiency?: number
  performance_ratio?: number
}

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit & { timeoutMs?: number }): Promise<T> {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), init?.timeoutMs ?? 30000)
  if (init?.signal) init.signal.addEventListener('abort', () => ctrl.abort())
  try {
    const res = await fetch(`${API_URL}${path}`, { ...init, signal: ctrl.signal })
    if (!res.ok) {
      let detail = res.statusText
      try {
        const body = await res.json()
        detail = body.detail ?? body.error ?? detail
      } catch {
        /* non-JSON error body */
      }
      throw new ApiError(detail, res.status)
    }
    return (await res.json()) as T
  } catch (e) {
    if (e instanceof ApiError) throw e
    if ((e as Error).name === 'AbortError') throw new ApiError('The request was cancelled or timed out.', 0)
    throw new ApiError('Cannot reach the analysis server. Is the backend running?', 0)
  } finally {
    clearTimeout(timer)
  }
}

export const api = {
  health: () => request<Record<string, unknown>>('/health', { timeoutMs: 8000 }),
  predict: (lat: number, lon: number, p: EnergyParams = {}, signal?: AbortSignal) =>
    request<Prediction>('/predict', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ lat, lon, ...p }),
      signal,
    }),
  compare: (sites: { lat: number; lon: number; name?: string }[], p: EnergyParams = {}) =>
    request<{ sites: CompareSiteResult[]; summary: { best_score: string | null; best_energy: string | null; n_ok: number } }>(
      '/compare',
      { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ sites, ...p }), timeoutMs: 60000 },
    ),
  layers: (scope: Scope) => request<{ scope: Scope; layers: LayerEntry[] }>(`/layers?scope=${scope}`),
  geocode: (q: string, signal?: AbortSignal) =>
    request<{ query: string; results: GeocodeResult[] }>(`/geocode?q=${encodeURIComponent(q)}&limit=5`, { signal, timeoutMs: 20000 }),
  famousPlaces: () => request<{ places: GeocodeResult[] }>('/places/famous', { timeoutMs: 8000 }),
  reportUrl: (lat: number, lon: number, p: EnergyParams = {}, name?: string) => {
    const q = new URLSearchParams()
    if (p.area_m2) q.set('area_m2', String(p.area_m2))
    if (p.efficiency) q.set('efficiency', String(p.efficiency))
    if (p.performance_ratio) q.set('performance_ratio', String(p.performance_ratio))
    if (name) q.set('name', name)
    return `${API_URL}/report/${lat.toFixed(5)}/${lon.toFixed(5)}?${q.toString()}`
  },
}

/* Static JSON shipped in /public/data (works without the backend). */
export async function staticJson<T>(file: string): Promise<T> {
  const res = await fetch(`/data/${file}`)
  if (!res.ok) throw new Error(`Missing /data/${file}`)
  return (await res.json()) as T
}
