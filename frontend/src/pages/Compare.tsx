import { ArrowSquareOut, Broadcast, GridFour, MapTrifold, Trash, Trophy, X } from '@phosphor-icons/react'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, LabelList, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Link } from 'react-router-dom'
import { ScoreGauge, SERIES } from '../components/charts'
import { ExcludedCard } from '../components/map/ResultPanel'
import { SearchBox } from '../components/SearchBox'
import { Bezel, ClassBadge, EASE, PillButton, Reveal, Skeleton } from '../components/ui'
import { api, type CompareSiteResult } from '../lib/api'
import { MAX_COMPARE, useCompareSites } from '../lib/compareStore'
import { ACRE_M2, coord, fmt } from '../lib/format'

const EXAMPLES = [
  { name: 'Bhadla Solar Park, India', lat: 27.5397, lon: 71.9157 },
  { name: 'Mojave Desert, USA', lat: 35.0, lon: -117.5 },
  { name: 'Brandenburg, Germany', lat: 52.3, lon: 13.6 },
]

const ROWS: { key: string; label: string; unit?: string }[] = [
  { key: 'ghi', label: 'Solar irradiance', unit: 'kWh/m²/day' },
  { key: 'cloud_cover', label: 'Cloud fraction', unit: '%' },
  { key: 'temperature', label: 'Air temperature', unit: '°C' },
  { key: 'slope', label: 'Slope', unit: '°' },
  { key: 'ndvi', label: 'NDVI' },
  { key: 'lst', label: 'Surface temperature', unit: '°C' },
  { key: 'landcover', label: 'Land cover (MODIS)' },
  { key: 'accessibility', label: 'Travel time to city', unit: 'min' },
]

const tooltipStyle = { background: 'var(--surface)', border: '1px solid var(--line)', borderRadius: 12, fontSize: 12, color: 'var(--ink)' }

export default function Compare() {
  const { sites, add, remove, clear } = useCompareSites()
  const [acres, setAcres] = useState(1)
  const [results, setResults] = useState<CompareSiteResult[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const key = sites.map((s) => s.id).join('|')
  useEffect(() => {
    if (sites.length < 2) {
      setResults(null)
      return
    }
    let cancelled = false
    setLoading(true)
    setError(null)
    const t = window.setTimeout(() => {
      api.compare(sites.map((s) => ({ lat: s.lat, lon: s.lon, name: s.name })), { area_m2: acres * ACRE_M2 })
        .then((r) => !cancelled && setResults(r.sites))
        .catch((e) => !cancelled && setError((e as Error).message))
        .finally(() => !cancelled && setLoading(false))
    }, 300)
    return () => {
      cancelled = true
      window.clearTimeout(t)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, acres])

  const ok = useMemo(() => (results ?? []).filter((r) => r.ok && r.result), [results])
  const scoreData = useMemo(() => {
    const metrics = [
      { k: 'suitability', label: 'Suitability' },
      { k: 'development', label: 'Development' },
      { k: 'ahp', label: 'AHP' },
    ] as const
    return metrics.map((m) => {
      const row: Record<string, string | number> = { metric: m.label }
      ok.forEach((r) => { row[r.name] = Math.round(r.result!.scores[m.k]?.score ?? 0) })
      return row
    })
  }, [ok])
  const energyData = useMemo(() => ok.map((r) => ({ name: r.name, mwh: Math.round(r.result!.energy?.annual_mwh ?? 0) })), [ok])

  return (
    <div className="mx-auto max-w-[1400px] px-4 pb-24 pt-32 md:px-8">
      <Reveal>
        <h1 className="text-4xl font-semibold tracking-tighter md:text-6xl">Compare up to three sites.</h1>
        <p className="mt-4 max-w-[56ch] text-lg leading-relaxed text-ink-2">Add places by search or from the map, then read their scores, energy and satellite profile side by side.</p>
      </Reveal>

      {/* picker */}
      <Reveal delay={0.08} className="mt-10">
        <Bezel radius="2rem" inner="grid gap-6 p-6 md:grid-cols-[1fr_auto] md:items-end md:p-7">
          <div className="space-y-4">
            <SearchBox
              placeholder={sites.length >= MAX_COMPARE ? 'Three sites queued. Remove one to add another.' : 'Add a site: search a city, region or landmark'}
              onSelect={(r) => add({ lat: r.lat, lon: r.lon, name: r.name })}
              className="max-w-xl"
            />
            <div className="flex flex-wrap items-center gap-2">
              <AnimatePresence initial={false}>
                {sites.map((s, i) => (
                  <motion.span
                    key={s.id}
                    layout
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.9 }}
                    transition={{ duration: 0.35, ease: EASE }}
                    className="inline-flex items-center gap-2 rounded-full bg-surface-2 py-1.5 pl-3 pr-1.5 text-sm ring-1 ring-line"
                  >
                    <span className="h-2.5 w-2.5 rounded-full" style={{ background: SERIES[i] }} aria-hidden />
                    <span className="max-w-[220px] truncate">{s.name}</span>
                    <button type="button" aria-label={`Remove ${s.name}`} onClick={() => remove(s.id)} className="flex h-6 w-6 items-center justify-center rounded-full text-muted hover:bg-ink/[0.07] hover:text-ink">
                      <X size={13} weight="light" />
                    </button>
                  </motion.span>
                ))}
              </AnimatePresence>
              {sites.length > 0 && (
                <button type="button" onClick={clear} className="inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs text-muted hover:text-ink">
                  <Trash size={14} weight="light" />Clear
                </button>
              )}
            </div>
          </div>
          <div className="flex flex-wrap items-end gap-3">
            <label className="block space-y-1.5">
              <span className="text-xs text-ink-2">Array area (acres)</span>
              <input
                type="number" min={0.1} max={50000} step={0.5} value={acres}
                onChange={(e) => { const v = Number(e.target.value); if (v > 0) setAcres(v) }}
                className="h-11 w-32 rounded-xl bg-surface-2 px-3 font-mono text-sm tabular ring-1 ring-line focus:outline-none focus:ring-accent/60"
              />
            </label>
            <PillButton to="/map" variant="ghost" icon={<MapTrifold size={16} weight="light" />}>Pick on map</PillButton>
          </div>
        </Bezel>
      </Reveal>

      {/* empty state */}
      {sites.length < 2 && (
        <Reveal delay={0.12} className="mt-16">
          <div className="grid items-center gap-10 md:grid-cols-[1.1fr_1fr]">
            <div>
              <h2 className="text-2xl font-semibold tracking-tight">{sites.length === 0 ? 'Nothing to compare yet.' : 'Add one more site.'}</h2>
              <p className="mt-3 max-w-[50ch] text-ink-2">
                Search above, or use Compare in any site analysis on the map. You can also start from three contrasting places.
              </p>
              <div className="mt-6">
                <PillButton onClick={() => { clear(); EXAMPLES.forEach((e) => add(e)) }}>Load an example</PillButton>
              </div>
            </div>
            <Bezel radius="1.75rem" inner="relative aspect-[16/9]">
              <img src="/img/layers/physical_global.webp" alt="Global suitability map" className="absolute inset-0 h-full w-full bg-surface-2 object-cover" loading="lazy" />
            </Bezel>
          </div>
        </Reveal>
      )}

      {error && <p className="mt-10 rounded-2xl bg-bad/10 px-4 py-3 text-sm ring-1 ring-bad/25">{error}</p>}

      {/* results */}
      {sites.length >= 2 && (
        <div className="mt-14 space-y-8">
          <div className={`grid gap-5 ${sites.length === 3 ? 'md:grid-cols-3' : 'md:grid-cols-2'}`}>
            {(loading || !results ? sites.map((s) => ({ index: -1, name: s.name, ok: false }) as CompareSiteResult) : results).map((r, i) => (
              <Reveal key={`${r.name}-${i}`} delay={i * 0.06}>
                <Bezel radius="2rem" className="h-full" inner="flex h-full flex-col p-6">
                  {loading || !results ? (
                    <div className="space-y-4">
                      <Skeleton className="h-5 w-2/3" /><Skeleton className="mx-auto h-36 w-36 rounded-full" /><Skeleton className="h-16 w-full" />
                    </div>
                  ) : !r.ok ? (
                    <div>
                      <p className="font-medium">{r.name}</p>
                      <p className="mt-3 text-sm text-bad">{r.error}</p>
                    </div>
                  ) : (
                    <SiteCard r={r} color={SERIES[i]} />
                  )}
                </Bezel>
              </Reveal>
            ))}
          </div>

          {ok.length >= 2 && !loading && (
            <div className="grid gap-5 lg:grid-cols-[1.4fr_1fr]">
              <Reveal>
                <Bezel radius="2rem" inner="p-6">
                  <p className="text-sm font-medium">Scores side by side</p>
                  <p className="text-xs text-muted">0-100. Suitability is physical; development follows historical build-out.</p>
                  <div className="mt-4 h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={scoreData} margin={{ top: 22, right: 8, left: -16, bottom: 0 }} barGap={3}>
                        <CartesianGrid vertical={false} stroke="var(--line)" />
                        <XAxis dataKey="metric" tickLine={false} axisLine={{ stroke: 'var(--line-strong)' }} tick={{ fill: 'var(--ink-2)', fontSize: 12 }} />
                        <YAxis domain={[0, 100]} tickLine={false} axisLine={false} tick={{ fill: 'var(--muted)', fontSize: 11 }} />
                        <Tooltip cursor={{ fill: 'var(--surface-2)' }} contentStyle={tooltipStyle} />
                        <Legend iconType="circle" itemSorter={null} wrapperStyle={{ fontSize: 12 }} />
                        {ok.map((r, i) => (
                          <Bar key={r.name} dataKey={r.name} fill={SERIES[results!.indexOf(r)] ?? SERIES[i]} radius={[4, 4, 0, 0]}>
                            <LabelList dataKey={r.name} position="top" fill="var(--ink-2)" fontSize={10.5} />
                          </Bar>
                        ))}
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </Bezel>
              </Reveal>
              <Reveal delay={0.06}>
                <Bezel radius="2rem" className="h-full" inner="h-full p-6">
                  <p className="text-sm font-medium">Annual energy</p>
                  <p className="text-xs text-muted">MWh per year from {fmt(acres, 2)} acre{acres === 1 ? '' : 's'} (20% modules, PR 0.75)</p>
                  <div className="mt-4 h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={energyData} layout="vertical" margin={{ top: 0, right: 48, left: 0, bottom: 0 }}>
                        <XAxis type="number" hide />
                        <YAxis type="category" dataKey="name" width={120} tickLine={false} axisLine={false} tick={{ fill: 'var(--ink-2)', fontSize: 11 }} />
                        <Tooltip cursor={{ fill: 'var(--surface-2)' }} contentStyle={tooltipStyle} formatter={(v) => [`${fmt(Number(v), 0)} MWh/yr`, 'Energy']} />
                        <Bar dataKey="mwh" radius={4} barSize={26}>
                          {energyData.map((d) => {
                            const idx = results!.findIndex((r) => r.name === d.name)
                            return <Cell key={d.name} fill={SERIES[idx]} />
                          })}
                          <LabelList dataKey="mwh" position="right" fill="var(--ink-2)" fontSize={11} />
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </Bezel>
              </Reveal>
            </div>
          )}

          {ok.length >= 2 && !loading && (
            <Reveal>
              <Bezel radius="2rem" inner="overflow-x-auto p-2">
                <table className="w-full min-w-[560px] text-sm">
                  <caption className="px-4 pb-2 pt-4 text-left text-sm font-medium">Remote-sensing profile</caption>
                  <thead>
                    <tr className="text-left text-xs text-muted">
                      <th className="px-4 py-3 font-medium">Variable</th>
                      {ok.map((r) => (
                        <th key={r.name} className="px-4 py-3 font-medium">
                          <span className="inline-flex items-center gap-2">
                            <span className="h-2 w-2 rounded-full" style={{ background: SERIES[results!.indexOf(r)] }} />
                            <span className="max-w-[160px] truncate">{r.name}</span>
                          </span>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {ROWS.map((row, i) => (
                      <tr key={row.key} className={i % 2 === 0 ? 'bg-surface-2/60' : ''}>
                        <td className="rounded-l-xl px-4 py-2.5 text-ink-2">{row.label}{row.unit ? <span className="ml-1 text-xs text-muted">({row.unit})</span> : null}</td>
                        {ok.map((r, j) => {
                          const v = r.result!.features[row.key]
                          return (
                            <td key={r.name} className={`px-4 py-2.5 ${j === ok.length - 1 ? 'rounded-r-xl' : ''}`}>
                              {typeof v === 'number' ? <span className="font-mono tabular">{fmt(v, 2)}</span> : v ?? 'n/a'}
                            </td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Bezel>
            </Reveal>
          )}
        </div>
      )}
    </div>
  )
}

function SiteCard({ r, color }: { r: CompareSiteResult; color: string }) {
  const p = r.result!
  const live = p.provenance.source === 'live_gee'
  return (
    <>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="flex items-center gap-2 font-medium">
            <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: color }} aria-hidden />
            <span className="truncate">{r.name}</span>
          </p>
          <p className="mt-1 font-mono text-xs text-muted tabular">{coord(p.lat, p.lon)}</p>
        </div>
        {r.rank === 1 && (
          <span className="inline-flex shrink-0 items-center gap-1 rounded-full bg-accent-soft px-2.5 py-1 text-[11px] font-medium text-accent-ink">
            <Trophy size={12} weight="fill" />Top suitability
          </span>
        )}
      </div>
      {p.excluded ? (
        <div className="mt-6"><ExcludedCard compact message={p.exclusion_message ?? `Excluded: ${p.excluded}.`} modelScore={p.model_score} /></div>
      ) : (
        <>
          <div className="mt-6 flex justify-center"><ScoreGauge score={p.score} size={148} /></div>
          <div className="mt-3 flex justify-center"><ClassBadge cls={p.class} /></div>
        </>
      )}
      <dl className="mt-6 grid grid-cols-3 gap-2 text-center">
        <div className="rounded-xl bg-surface-2 py-2.5 ring-1 ring-line"><dt className="text-[10.5px] text-muted">Development</dt><dd className="font-mono text-lg tabular">{Math.round(p.scores.development.score)}</dd></div>
        <div className="rounded-xl bg-surface-2 py-2.5 ring-1 ring-line"><dt className="text-[10.5px] text-muted">AHP</dt><dd className="font-mono text-lg tabular">{Math.round(p.scores.ahp?.score ?? 0)}</dd></div>
        <div className="rounded-xl bg-surface-2 py-2.5 ring-1 ring-line"><dt className="text-[10.5px] text-muted">MWh/yr</dt><dd className="font-mono text-lg tabular">{fmt(p.energy?.annual_mwh ?? 0, 0)}</dd></div>
      </dl>
      <div className="mt-auto flex items-center justify-between pt-5 text-xs text-muted">
        <span className="inline-flex items-center gap-1.5">
          {live ? <Broadcast size={14} weight="light" className="text-accent" /> : <GridFour size={14} weight="light" />}
          {live ? 'Live Earth Engine' : `Grid estimate, ${fmt(p.provenance.cell_distance_km ?? 0)} km`}
        </span>
        <Link to={`/map?lat=${p.lat}&lon=${p.lon}`} className="inline-flex items-center gap-1 hover:text-ink">
          Open <ArrowSquareOut size={13} weight="light" />
        </Link>
      </div>
    </>
  )
}
