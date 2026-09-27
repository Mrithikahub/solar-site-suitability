import L from 'leaflet'
import { useEffect, useMemo, useState } from 'react'
import { ImageOverlay, MapContainer, TileLayer } from 'react-leaflet'
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { staticJson, type LayerEntry } from '../lib/api'
import { basemap, ESRI_ATTRIBUTION } from '../lib/basemaps'
import { useTheme } from '../lib/theme'
import { RampLegend } from './charts'
import { Bezel } from './ui'

/* eslint-disable @typescript-eslint/no-explicit-any */
type ManifestEntry = Omit<LayerEntry, 'name' | 'scope' | 'image_url'>

const VIEWS: { key: string; label: string; img: string }[] = [
  { key: 'truecolor', label: 'True colour', img: 'truecolor' },
  { key: 'ndvi', label: 'NDVI', img: 'ndvi' },
  { key: 'ndbi', label: 'NDBI', img: 'ndbi' },
  { key: 'lst', label: 'LST', img: 'lst' },
  { key: 'slope', label: 'Slope', img: 'slope' },
  { key: 'landcover', label: 'Land cover', img: 'landcover' },
  { key: 'ghi', label: 'GHI', img: 'tn_ghi' },
  { key: 'suitability', label: 'ML suitability', img: 'ml_tn' },
  { key: 'ahp', label: 'AHP', img: 'ahp_tn' },
]

const LEAK = [
  { index: 'NDVI', solarPre: 0.29, solarNow: 0.2, otherPre: 0.44, otherNow: 0.45 },
  { index: 'NDBI', solarPre: 0.05, solarNow: 0.25, otherPre: -0.03, otherNow: -0.01 },
  { index: 'LST (°C)', solarPre: 45.6, solarNow: 44.5, otherPre: 40.0, otherNow: 38.4 },
]

const SOURCES = [
  ['Sentinel-2 MSI L2A', '10-20 m', 'NDVI, NDBI and true colour; Cloud Score+ masked 12-month median of 2,086 scenes.'],
  ['Landsat 8/9 TIRS', '100 m (30 m product)', 'Land surface temperature from ST_B10; 2013-15 scenes give the pre-construction baseline.'],
  ['SRTM GL1', '30 m', 'Elevation, slope and aspect.'],
  ['ESA WorldCover', '10 m', 'Land cover (2021) for exclusions and display.'],
  ['NASA POWER', '0.5-1°', 'GHI, air temperature and cloud cover (2005-2024 climatology).'],
  ['OpenStreetMap', 'vector', '344 solar sites, 161,768 road segments, 6,371 power lines, 2,710 substations.'],
]

/**
 * High-resolution regional case study (Tamil Nadu, India), shown inside the
 * Methodology page: layer viewer, local-vs-global comparison, 10 m leakage.
 */
export function CaseStudy({ ahpTable }: { ahpTable?: React.ReactNode }) {
  const { theme } = useTheme()
  const [layers, setLayers] = useState<Record<string, ManifestEntry> | null>(null)
  const [view, setView] = useState('suitability')
  const [m, setM] = useState<Record<string, any> | null>(null)

  useEffect(() => {
    staticJson<Record<string, Record<string, ManifestEntry>>>('layers_manifest.json').then((x) => setLayers(x.tamil_nadu)).catch(() => setLayers({}))
    staticJson<Record<string, any>>('metrics.json').then(setM).catch(() => setM(null))
  }, [])

  const gvl = m
    ? [
        { name: 'Regional model (RF)', auc: m.tamil_nadu.global_vs_local.local.roc_auc },
        { name: 'Global model', auc: m.tamil_nadu.global_vs_local.global.roc_auc },
        { name: 'AHP overlay', auc: m.tamil_nadu.global_vs_local.ahp.roc_auc },
      ]
    : null
  const active = layers?.[view]
  const v = VIEWS.find((x) => x.key === view)!
  const bounds = useMemo<L.LatLngBoundsExpression | null>(() => {
    const b = layers?.ndvi?.bounds
    return b ? [[b.south, b.west], [b.north, b.east]] : null
  }, [layers])
  const base = basemap('map', theme).base

  return (
    <div className="space-y-12">
      <p className="max-w-[68ch] leading-relaxed text-ink-2">
        Where finer data exists, the same pipeline runs at 10-30 m with local grid infrastructure. The reference region is Tamil Nadu,
        India (130,500 km², 344 mapped solar sites, 4,420 labelled points), with Sentinel-2, Landsat thermal, SRTM terrain and
        OpenStreetMap roads, power lines and substations. On the map these appear as <span className="text-ink">High-res layers</span>.
      </p>

      {/* layer viewer */}
      <Bezel radius="1.75rem" pad="0.5rem" inner="flex flex-col">
        <div className="relative h-[58dvh] min-h-[380px]">
          {bounds && (
            <MapContainer bounds={bounds} scrollWheelZoom={false} className="h-full w-full" minZoom={6} maxZoom={12} maxBounds={[[6.5, 74.5], [15, 82]]}>
              <TileLayer key={base} url={base} attribution={ESRI_ATTRIBUTION} />
              <ImageOverlay key={view} url={`/img/layers/${v.img}.webp`} bounds={bounds} opacity={view === 'truecolor' ? 1 : 0.88} />
            </MapContainer>
          )}
        </div>
        <div className="space-y-4 border-t border-line p-4">
          <div className="-mx-1 flex gap-1.5 overflow-x-auto px-1 pb-1" role="tablist" aria-label="High-resolution layers">
            {VIEWS.map((x) => (
              <button
                key={x.key}
                role="tab"
                aria-selected={view === x.key}
                type="button"
                onClick={() => setView(x.key)}
                className={`shrink-0 rounded-full px-3 py-1.5 text-xs transition-all duration-300 active:scale-95 ${view === x.key ? 'bg-ink text-bg' : 'bg-ink/[0.05] text-ink-2 ring-1 ring-line hover:text-ink dark:bg-white/[0.05]'}`}
              >
                {x.label}
              </button>
            ))}
          </div>
          {active && active.kind === 'continuous' && (
            <RampLegend palette={active.palette ?? []} min={active.min ?? 0} max={active.max ?? 1} unit={active.unit} title={active.title} />
          )}
          {active && active.kind === 'categorical' && (
            <div className="flex flex-wrap gap-x-4 gap-y-1.5">
              {active.classes?.map((c) => (
                <span key={c.code} className="flex items-center gap-1.5 text-[11px] text-ink-2">
                  <span className="h-2.5 w-2.5 rounded-[3px]" style={{ background: c.color }} />{c.name}
                </span>
              ))}
            </div>
          )}
          {active && <p className="text-[11px] text-muted">{active.title}. Source: {active.source}</p>}
        </div>
      </Bezel>

      {/* regional vs global */}
      <div className="grid items-center gap-8 lg:grid-cols-2">
        <div>
          <h3 className="text-xl font-semibold tracking-tight">Regional model vs global model</h3>
          <p className="mt-3 max-w-[56ch] leading-relaxed text-ink-2">
            All three methods are scored on the same held-out sites, split by solar park so no park appears in both training and test data.
            The regional model sees 30 m Landsat and exact distances to substations; the global model only has 500 m MODIS and travel time to
            cities, and has never seen these labels, yet transfers well.
          </p>
        </div>
        <Bezel radius="1.75rem" inner="p-6">
          <p className="text-sm font-medium">ROC-AUC on the regional test split</p>
          <p className="text-xs text-muted">1.0 is a perfect ranking of plant vs non-plant sites; 0.5 is chance.</p>
          <div className="mt-4 h-56">
            {gvl ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={gvl} layout="vertical" margin={{ top: 0, right: 48, left: 0, bottom: 0 }}>
                  <CartesianGrid horizontal={false} stroke="var(--line)" />
                  <XAxis type="number" domain={[0.5, 1]} tickLine={false} axisLine={false} tick={{ fill: 'var(--muted)', fontSize: 11 }} />
                  <YAxis type="category" dataKey="name" width={132} tickLine={false} axisLine={false} tick={{ fill: 'var(--ink-2)', fontSize: 12 }} />
                  <Tooltip cursor={{ fill: 'var(--surface-2)' }} contentStyle={{ background: 'var(--surface)', border: '1px solid var(--line)', borderRadius: 12, fontSize: 12 }} formatter={(x) => Number(x).toFixed(3)} />
                  <Bar dataKey="auc" radius={4} barSize={26}>
                    {gvl.map((d, i) => <Cell key={d.name} fill={['#1baf7a', '#2a78d6', '#eb6834'][i]} />)}
                    <LabelList dataKey="auc" position="right" formatter={(x: unknown) => Number(x).toFixed(3)} fill="var(--ink-2)" fontSize={11} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : <div className="skeleton h-full" />}
          </div>
        </Bezel>
      </div>

      {/* leakage at 10 m */}
      <div>
        <h3 className="text-xl font-semibold tracking-tight">At 10 m, the satellite sees the panels</h3>
        <p className="mt-3 max-w-[64ch] leading-relaxed text-ink-2">
          Median index values at 1,768 plant points and 2,652 non-plant points. NDBI at plants jumps from 0.05 to 0.25 after construction,
          while other land barely moves, so the regional model is trained on 2013-15 Landsat values.
        </p>
        <div className="mt-6 grid gap-4 md:grid-cols-3">
          {LEAK.map((row) => (
            <div key={row.index} className="rounded-2xl bg-surface-2 p-5 ring-1 ring-line">
              <p className="text-sm font-medium">{row.index}</p>
              <div className="mt-4 grid grid-cols-2 gap-4">
                <div>
                  <p className="text-[11px] text-muted">Solar plants</p>
                  <p className="mt-1 font-mono text-xl tabular">{row.solarPre} <span className="text-muted">to</span> <span className="text-accent-ink">{row.solarNow}</span></p>
                </div>
                <div>
                  <p className="text-[11px] text-muted">Other land</p>
                  <p className="mt-1 font-mono text-xl tabular">{row.otherPre} <span className="text-muted">to</span> {row.otherNow}</p>
                </div>
              </div>
              <p className="mt-3 text-[11px] text-muted">2013-15 baseline to Sep 2025 - Sep 2026</p>
            </div>
          ))}
        </div>
      </div>

      {/* AHP for the region (passed in so the Methodology page owns the table component) */}
      {ahpTable && (
        <div>
          <h3 className="text-xl font-semibold tracking-tight">Regional AHP criteria</h3>
          <p className="mt-3 max-w-[64ch] leading-relaxed text-ink-2">Seven criteria, including distance to substations and roads from OpenStreetMap.</p>
          {ahpTable}
        </div>
      )}

      {/* explanations */}
      <div>
        <h3 className="text-xl font-semibold tracking-tight">What drives the regional model</h3>
        <div className="mt-6 grid gap-5 lg:grid-cols-2">
          <Bezel radius="1.5rem" inner="bg-[#fcfcfb] p-2"><img src="/figures/shap_summary_tamil_nadu.webp" alt="SHAP summary plot for the regional model" loading="lazy" className="w-full" /></Bezel>
          <Bezel radius="1.5rem" inner="bg-[#fcfcfb] p-2"><img src="/figures/feature_importance_tamil_nadu.webp" alt="Feature importance of the regional Random Forest" loading="lazy" className="w-full" /></Bezel>
        </div>
        <p className="mt-4 max-w-[70ch] text-sm text-ink-2">
          Land surface temperature and distance to substations lead: parks cluster along the transmission network on hot, low-NDVI land
          that has little agricultural value.
        </p>
      </div>

      {/* sources */}
      <div>
        <h3 className="text-xl font-semibold tracking-tight">High-resolution data</h3>
        <div className="mt-6 grid gap-x-10 gap-y-6 sm:grid-cols-2">
          {SOURCES.map(([name, res, body]) => (
            <div key={name} className="border-t border-line pt-4">
              <p className="flex items-baseline justify-between gap-3"><span className="font-medium">{name}</span><span className="font-mono text-xs text-muted">{res}</span></p>
              <p className="mt-2 text-sm leading-relaxed text-ink-2">{body}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
