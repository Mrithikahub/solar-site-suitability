import L from 'leaflet'
import { useEffect, useMemo, useState } from 'react'
import { ImageOverlay, MapContainer, TileLayer } from 'react-leaflet'
import { Bar, BarChart, CartesianGrid, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis, Cell } from 'recharts'
import { RampLegend } from '../components/charts'
import { Bezel, PillButton, Reveal } from '../components/ui'
import { staticJson, type LayerEntry } from '../lib/api'
import { basemap, ESRI_ATTRIBUTION } from '../lib/basemaps'
import { useTheme } from '../lib/theme'

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
  ['Sentinel-2 MSI L2A', '10-20 m', 'NDVI, NDBI, true colour. Cloud Score+ masked, 12-month median.'],
  ['Landsat 8/9 TIRS', '100 m (30 m product)', 'Land surface temperature from ST_B10; 2013-15 baseline for training.'],
  ['SRTM GL1', '30 m', 'Elevation, slope and aspect.'],
  ['ESA WorldCover', '10 m', 'Land cover (2021), used for exclusions and display.'],
  ['NASA POWER', '0.5-1°', 'GHI, air temperature, cloud cover (2005-2024 climatology).'],
  ['OpenStreetMap', 'vector', '344 solar sites, 161,768 road segments, 2,710 substations.'],
]

export default function TamilNadu() {
  const { theme } = useTheme()
  const [layers, setLayers] = useState<Record<string, Omit<LayerEntry, 'name' | 'scope' | 'image_url'>> | null>(null)
  const [view, setView] = useState('suitability')
  const [gvl, setGvl] = useState<{ name: string; auc: number; f1: number }[] | null>(null)

  useEffect(() => {
    staticJson<Record<string, Record<string, Omit<LayerEntry, 'name' | 'scope' | 'image_url'>>>>('layers_manifest.json').then((m) => setLayers(m.tamil_nadu)).catch(() => setLayers({}))
    staticJson<Record<string, any>>('metrics.json').then((m) => {
      const g = m.tamil_nadu.global_vs_local
      setGvl([
        { name: 'Local model (RF)', auc: g.local.roc_auc, f1: g.local.f1 },
        { name: 'Global model', auc: g.global.roc_auc, f1: g.global.f1 },
        { name: 'AHP overlay', auc: g.ahp.roc_auc, f1: g.ahp.f1 },
      ])
    }).catch(() => setGvl(null))
  }, [])

  const active = layers?.[view]
  const v = VIEWS.find((x) => x.key === view)!
  const bounds = useMemo<L.LatLngBoundsExpression | null>(() => {
    const b = layers?.ndvi?.bounds
    return b ? [[b.south, b.west], [b.north, b.east]] : null
  }, [layers])
  const base = basemap('map', theme).base

  return (
    <div className="pb-8 pt-28">
      {/* hero: text + live layer viewer */}
      <section className="mx-auto grid max-w-[1400px] gap-10 px-4 md:grid-cols-[0.9fr_1.1fr] md:px-8">
        <Reveal className="flex flex-col justify-center">
          <p className="inline-flex w-max rounded-full bg-accent-soft px-3 py-1 text-[11px] font-medium uppercase tracking-[0.18em] text-accent-ink">Case study</p>
          <h1 className="mt-5 text-5xl font-semibold leading-[1.02] tracking-tighter md:text-6xl">Tamil Nadu at 285 metres.</h1>
          <p className="mt-5 max-w-[48ch] text-lg leading-relaxed text-ink-2">
            India's solar leader, mapped with 10 m Sentinel-2, Landsat thermal, SRTM terrain and OpenStreetMap grid data.
          </p>
          <dl className="mt-10 grid max-w-md grid-cols-3 gap-6">
            <div><dt className="text-xs text-muted">Solar sites</dt><dd className="font-mono text-3xl tabular">344</dd></div>
            <div><dt className="text-xs text-muted">Labelled points</dt><dd className="font-mono text-3xl tabular">4,420</dd></div>
            <div><dt className="text-xs text-muted">Local ROC-AUC</dt><dd className="font-mono text-3xl tabular">{gvl ? gvl[0].auc.toFixed(3) : '0.931'}</dd></div>
          </dl>
          <div className="mt-10"><PillButton to="/map">Open the map</PillButton></div>
        </Reveal>

        <Reveal delay={0.1}>
          <Bezel radius="2rem" pad="0.5rem" inner="flex flex-col">
            <div className="relative h-[62dvh] min-h-[420px]">
              {bounds && (
                <MapContainer bounds={bounds} scrollWheelZoom={false} className="h-full w-full" zoomControl minZoom={6} maxZoom={12} maxBounds={[[6.5, 74.5], [15, 82]]}>
                  <TileLayer key={base} url={base} attribution={ESRI_ATTRIBUTION} />
                  <ImageOverlay key={view} url={`/img/layers/${v.img}.webp`} bounds={bounds} opacity={view === 'truecolor' ? 1 : 0.88} />
                </MapContainer>
              )}
            </div>
            <div className="space-y-4 border-t border-line p-4">
              <div className="-mx-1 flex gap-1.5 overflow-x-auto px-1 pb-1" role="tablist" aria-label="Case-study layers">
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
        </Reveal>
      </section>

      {/* local vs global */}
      <section className="mx-auto mt-28 grid max-w-[1400px] items-center gap-12 px-4 md:grid-cols-2 md:px-8">
        <Reveal>
          <h2 className="text-4xl font-semibold leading-[1.05] tracking-tighter md:text-5xl">A local model beats a global one, on local ground.</h2>
          <p className="mt-6 max-w-[52ch] leading-relaxed text-ink-2">
            All three methods were scored on the same held-out Tamil Nadu sites, split by solar park so no park appears in both training and test data.
            The local model sees 30 m Landsat and exact distances to substations; the global model only has 500 m MODIS and travel time to cities, yet still generalises well.
          </p>
        </Reveal>
        <Reveal delay={0.08}>
          <Bezel radius="2rem" inner="p-6">
            <p className="text-sm font-medium">ROC-AUC on the Tamil Nadu test split</p>
            <p className="text-xs text-muted">1.0 is a perfect ranking of plant vs non-plant sites; 0.5 is chance.</p>
            <div className="mt-4 h-64">
              {gvl ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={gvl} layout="vertical" margin={{ top: 0, right: 48, left: 0, bottom: 0 }}>
                    <CartesianGrid horizontal={false} stroke="var(--line)" />
                    <XAxis type="number" domain={[0.5, 1]} tickLine={false} axisLine={false} tick={{ fill: 'var(--muted)', fontSize: 11 }} />
                    <YAxis type="category" dataKey="name" width={122} tickLine={false} axisLine={false} tick={{ fill: 'var(--ink-2)', fontSize: 12 }} />
                    <Tooltip cursor={{ fill: 'var(--surface-2)' }} contentStyle={{ background: 'var(--surface)', border: '1px solid var(--line)', borderRadius: 12, fontSize: 12 }} formatter={(x) => Number(x).toFixed(3)} />
                    <Bar dataKey="auc" radius={4} barSize={28}>
                      {gvl.map((d, i) => <Cell key={d.name} fill={['#1baf7a', '#2a78d6', '#eb6834'][i]} />)}
                      <LabelList dataKey="auc" position="right" formatter={(x: unknown) => Number(x).toFixed(3)} fill="var(--ink-2)" fontSize={11} />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : <div className="skeleton h-full" />}
            </div>
          </Bezel>
        </Reveal>
      </section>

      {/* leakage at 10 m */}
      <section className="mx-auto mt-28 max-w-[1400px] px-4 md:px-8">
        <Reveal>
          <h2 className="max-w-3xl text-4xl font-semibold leading-[1.05] tracking-tighter md:text-5xl">At 10 metres, the satellite sees the panels.</h2>
          <p className="mt-5 max-w-[62ch] leading-relaxed text-ink-2">
            Median index values at 1,768 plant points and 2,652 non-plant points. NDBI at plants jumps from 0.05 to 0.25 after construction,
            while non-plant land barely moves. The local model is therefore trained on 2013-15 Landsat values.
          </p>
        </Reveal>
        <div className="mt-12 grid gap-5 md:grid-cols-3">
          {LEAK.map((row, i) => (
            <Reveal key={row.index} delay={i * 0.06}>
              <Bezel radius="1.75rem" className="h-full" inner="h-full p-6">
                <p className="text-sm font-medium">{row.index}</p>
                <div className="mt-5 grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-[11px] text-muted">Solar plants</p>
                    <p className="mt-1 font-mono text-2xl tabular">{row.solarPre} <span className="text-muted">to</span> <span className="text-accent-ink">{row.solarNow}</span></p>
                  </div>
                  <div>
                    <p className="text-[11px] text-muted">Other land</p>
                    <p className="mt-1 font-mono text-2xl tabular">{row.otherPre} <span className="text-muted">to</span> {row.otherNow}</p>
                  </div>
                </div>
                <p className="mt-4 text-[11px] text-muted">2013-15 baseline to Sep 2025 - Sep 2026</p>
              </Bezel>
            </Reveal>
          ))}
        </div>
      </section>

      {/* explanations */}
      <section className="mx-auto mt-28 max-w-[1400px] px-4 md:px-8">
        <Reveal>
          <h2 className="text-4xl font-semibold tracking-tighter md:text-5xl">What drives the local model.</h2>
        </Reveal>
        <div className="mt-10 grid gap-5 lg:grid-cols-2">
          <Reveal><Bezel radius="1.75rem" inner="bg-[#fcfcfb] p-3"><img src="/figures/shap_summary_tamil_nadu.webp" alt="SHAP summary plot for the Tamil Nadu model" loading="lazy" className="w-full" /></Bezel></Reveal>
          <Reveal delay={0.06}><Bezel radius="1.75rem" inner="bg-[#fcfcfb] p-3"><img src="/figures/feature_importance_tamil_nadu.webp" alt="Feature importance of the Tamil Nadu Random Forest" loading="lazy" className="w-full" /></Bezel></Reveal>
        </div>
        <p className="mt-4 max-w-[70ch] text-sm text-ink-2">
          Distance to substations and power lines dominates: Tamil Nadu parks cluster along the transmission network in the dry south.
          Low pre-construction NDVI and high land surface temperature mark the barren, low-value land developers prefer.
        </p>
      </section>

      {/* sources */}
      <section className="mx-auto mt-28 max-w-[1400px] px-4 md:px-8">
        <Reveal>
          <h2 className="text-3xl font-semibold tracking-tighter md:text-4xl">Case-study data.</h2>
        </Reveal>
        <div className="mt-8 grid gap-x-10 gap-y-6 sm:grid-cols-2 lg:grid-cols-3">
          {SOURCES.map(([name, res, body], i) => (
            <Reveal key={name} delay={i * 0.04} className="border-t border-line pt-4">
              <p className="flex items-baseline justify-between gap-3"><span className="font-medium">{name}</span><span className="font-mono text-xs text-muted">{res}</span></p>
              <p className="mt-2 text-sm leading-relaxed text-ink-2">{body}</p>
            </Reveal>
          ))}
        </div>
      </section>
    </div>
  )
}
