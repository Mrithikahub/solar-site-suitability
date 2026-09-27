import {
  ArrowClockwise, Broadcast, CheckCircle, CloudSun, DownloadSimple, GridFour, Leaf, Lightning, Plus, Prohibit, X,
} from '@phosphor-icons/react'
import { useState } from 'react'
import type { Prediction } from '../../lib/api'
import { api } from '../../lib/api'
import { useCompareSites } from '../../lib/compareStore'
import { ACRE_M2, coord, fmt } from '../../lib/format'
import { ScoreGauge, ShapChart } from '../charts'
import { ClassBadge, cx, IconButton, PillButton, Skeleton } from '../ui'

export interface EnergyInputs {
  acres: number
  efficiency: number // percent
  pr: number // percent
}

const FEATURE_ORDER = ['ghi', 'cloud_cover', 'temperature', 'slope', 'elevation', 'ndvi', 'ndbi', 'lst', 'landcover', 'worldcover', 'accessibility', 'nightlights', 'population']

function Section({ title, children, className }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <section className={cx('border-t border-line px-6 py-6', className)}>
      <h3 className="mb-4 text-[13px] font-medium text-ink-2">{title}</h3>
      {children}
    </section>
  )
}

function Provenance({ p }: { p: Prediction['provenance'] }) {
  const live = p.source === 'live_gee'
  return (
    <div className={cx('flex items-start gap-2.5 rounded-2xl px-3.5 py-2.5 text-xs ring-1', live ? 'bg-accent-soft text-ink ring-accent/25' : 'bg-ink/[0.04] text-ink-2 ring-line dark:bg-white/[0.04]')}>
      {live ? <Broadcast size={16} weight="light" className="mt-px shrink-0 text-accent" /> : <GridFour size={16} weight="light" className="mt-px shrink-0" />}
      <span>
        <span className="font-medium text-ink">{live ? 'Live Earth Engine extraction' : 'Grid estimate'}</span>
        {live
          ? <> from satellite data at this exact point{p.latency_s ? `, ${p.latency_s}s` : ''}{p.cached ? ' (cached)' : ''}.</>
          : <> from the nearest 0.5° cell, {fmt(p.cell_distance_km ?? 0)} km away. Live extraction was unavailable, so treat values as regional.</>}
      </span>
    </div>
  )
}

function EnergyBlock({ pred, inputs, setInputs }: { pred: Prediction; inputs: EnergyInputs; setInputs: (e: EnergyInputs) => void }) {
  const e = pred.energy
  if (!e) return <p className="text-sm text-muted">No irradiance data at this location.</p>
  const field = (key: keyof EnergyInputs, label: string, min: number, max: number, step: number, suffix: string) => (
    <label className="block space-y-1.5">
      <span className="text-[11px] text-muted">{label}</span>
      <span className="flex items-center rounded-xl bg-surface-2 px-3 ring-1 ring-line focus-within:ring-accent/60">
        <input
          type="number"
          inputMode="decimal"
          min={min}
          max={max}
          step={step}
          value={inputs[key]}
          onChange={(ev) => {
            const v = Number(ev.target.value)
            if (!Number.isNaN(v)) setInputs({ ...inputs, [key]: Math.min(max, Math.max(min, v)) })
          }}
          className="h-9 w-full min-w-0 bg-transparent font-mono text-sm text-ink tabular focus:outline-none"
        />
        <span className="text-xs text-muted">{suffix}</span>
      </span>
    </label>
  )
  return (
    <div className="space-y-5">
      <div className="flex items-end justify-between gap-4">
        <div>
          <p className="font-mono text-4xl font-medium tracking-tight tabular">{fmt(e.annual_mwh, 0)}</p>
          <p className="mt-1 text-xs text-muted">MWh per year from {fmt(e.area_acres, 2)} acre{e.area_acres === 1 ? '' : 's'}</p>
        </div>
        <Lightning size={28} weight="light" className="text-accent" />
      </div>
      <div className="grid grid-cols-3 gap-2.5">
        {field('acres', 'Area', 0.1, 50000, 0.5, 'acre')}
        {field('efficiency', 'Module η', 5, 40, 0.5, '%')}
        {field('pr', 'Perf. ratio', 30, 100, 1, '%')}
      </div>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
        <div><dt className="text-[11px] text-muted">Peak capacity</dt><dd className="font-mono tabular">{fmt(e.peak_capacity_kwp, 0)} kWp</dd></div>
        <div><dt className="text-[11px] text-muted">Specific yield</dt><dd className="font-mono tabular">{fmt(e.specific_yield_kwh_per_kwp, 0)} kWh/kWp</dd></div>
        <div><dt className="text-[11px] text-muted">CO₂ avoided</dt><dd className="font-mono tabular">{fmt(e.co2_avoided_tonnes, 0)} t/yr</dd></div>
        <div><dt className="text-[11px] text-muted">Households supplied</dt><dd className="font-mono tabular">{fmt(e.households_powered, 0)}</dd></div>
      </dl>
      <p className="font-mono text-[11px] text-muted">
        E = A × GHI × 365 × η × PR = {fmt(e.area_m2, 0)} m² × {e.ghi_kwh_m2_day.toFixed(2)} × 365 × {e.efficiency.toFixed(2)} × {e.performance_ratio.toFixed(2)}
      </p>
    </div>
  )
}

/** Shown instead of a bare 0 when the exclusion mask removes the site. */
export function ExcludedCard({ message, modelScore, compact }: { message: string; modelScore: number; compact?: boolean }) {
  const [head, ...rest] = message.replace(/\.$/, '').split(', ')
  return (
    <div className={cx('rounded-2xl bg-bad/[0.08] ring-1 ring-bad/25', compact ? 'p-4' : 'p-5')} role="status">
      <div className="flex items-start gap-3.5">
        <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-bad/15">
          <Prohibit size={22} weight="light" className="text-bad" />
        </span>
        <div className="min-w-0">
          <p className="text-[15px] font-semibold text-ink">{head}</p>
          {rest.length > 0 && <p className="mt-0.5 text-[13px] leading-snug text-ink-2">{rest.join(', ')}.</p>}
          <p className="mt-2.5 text-[11.5px] leading-snug text-muted">
            Suitability is set to 0 by the exclusion mask. Without it, the model would score this land {fmt(modelScore, 0)}.
          </p>
        </div>
      </div>
    </div>
  )
}

export function ResultSkeleton() {
  return (
    <div className="space-y-6 p-6" aria-busy aria-label="Analysing location">
      <Skeleton className="h-4 w-40" />
      <Skeleton className="h-11 w-full rounded-2xl" />
      <div className="flex items-center gap-6">
        <Skeleton className="h-40 w-40 rounded-full" />
        <div className="flex-1 space-y-3"><Skeleton className="h-5 w-24" /><Skeleton className="h-4 w-full" /><Skeleton className="h-4 w-4/5" /></div>
      </div>
      <div className="grid grid-cols-2 gap-3"><Skeleton className="h-28" /><Skeleton className="h-28" /></div>
      <Skeleton className="h-44 w-full" />
      <p className="text-xs text-muted">Reading satellite layers for this point…</p>
    </div>
  )
}

export function ResultPanel({ pred, loading, error, point, inputs, setInputs, onClose, onRetry }: {
  pred: Prediction | null
  loading: boolean
  error: string | null
  point: { lat: number; lon: number } | null
  inputs: EnergyInputs
  setInputs: (e: EnergyInputs) => void
  onClose: () => void
  onRetry: () => void
}) {
  const { add, has } = useCompareSites()
  const [toast, setToast] = useState<string | null>(null)
  const params = { area_m2: inputs.acres * ACRE_M2, efficiency: inputs.efficiency / 100, performance_ratio: inputs.pr / 100 }

  const addCompare = () => {
    if (!pred) return
    const r = add({ lat: pred.lat, lon: pred.lon, name: `Site ${coord(pred.lat, pred.lon)}` })
    setToast(r === 'added' ? 'Added to compare' : r === 'full' ? 'Compare holds 3 sites. Remove one first.' : 'Already in compare')
    window.setTimeout(() => setToast(null), 2600)
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between gap-3 px-6 pb-3 pt-5">
        <div className="min-w-0">
          <p className="text-[11px] text-muted">Site analysis</p>
          <p className="truncate font-mono text-sm tabular">{point ? coord(point.lat, point.lon) : ''}</p>
        </div>
        <IconButton label="Close analysis" onClick={onClose}><X size={18} weight="light" /></IconButton>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
        {loading && <ResultSkeleton />}

        {!loading && error && (
          <div className="space-y-4 p-6">
            <div className="flex items-start gap-3 rounded-2xl bg-bad/10 px-4 py-3 text-sm ring-1 ring-bad/25">
              <Prohibit size={18} weight="light" className="mt-0.5 shrink-0 text-bad" />
              <p className="text-ink">{error}</p>
            </div>
            <PillButton variant="ghost" onClick={onRetry} icon={<ArrowClockwise size={16} weight="light" />}>Try again</PillButton>
          </div>
        )}

        {!loading && !error && pred && (
          <>
            <div className="space-y-5 px-6 pb-6">
              <Provenance p={pred.provenance} />
              {pred.excluded ? (
                <ExcludedCard message={pred.exclusion_message ?? `Excluded: ${pred.excluded}.`} modelScore={pred.model_score} />
              ) : (
                <div className="flex items-center gap-5">
                  <ScoreGauge score={pred.score} size={156} />
                  <div className="min-w-0 space-y-2">
                    <ClassBadge cls={pred.class} />
                    <p className="text-[13px] leading-relaxed text-ink-2">{pred.scores.suitability.description}</p>
                  </div>
                </div>
              )}
              <div className="grid grid-cols-2 gap-3">
                {[pred.scores.development, pred.scores.ahp].filter(Boolean).map((s) => (
                  <div key={s!.label} className="rounded-2xl bg-surface-2 p-4 ring-1 ring-line">
                    <p className="text-[12px] font-medium text-ink-2">{s!.label}</p>
                    <div className="mt-2 flex items-baseline justify-between">
                      <span className="font-mono text-2xl font-medium tabular">{Math.round(s!.score)}</span>
                      <ClassBadge cls={s!.class} size="sm" />
                    </div>
                    <p className="mt-2 text-[11.5px] leading-snug text-muted">{s!.description}</p>
                  </div>
                ))}
              </div>
              {pred.tamil_nadu_local && (
                <div className="flex items-center justify-between rounded-2xl bg-surface-2 px-4 py-3 ring-1 ring-line">
                  <div>
                    <p className="text-[12px] font-medium text-ink-2">{pred.tamil_nadu_local.label ?? 'High-res regional model'}</p>
                    <p className="text-[11.5px] text-muted">10-30 m imagery with local grid distances</p>
                  </div>
                  <div className="text-right">
                    <p className="font-mono text-xl tabular">{Math.round(pred.tamil_nadu_local.score)}</p>
                    <ClassBadge cls={pred.tamil_nadu_local.class} size="sm" />
                  </div>
                </div>
              )}
            </div>

            <Section title="Energy estimate">
              <EnergyBlock pred={pred} inputs={inputs} setInputs={setInputs} />
            </Section>

            <Section title="Why this score">
              <p className="-mt-2 mb-3 text-[11.5px] leading-snug text-muted">
                SHAP contributions to suitability, starting from the model average of {fmt(pred.shap.base_score)}. Red raises the score, blue lowers it.
              </p>
              <ShapChart items={pred.shap.contributions} max={8} />
            </Section>

            <Section title="Remote-sensing profile">
              <dl className="grid grid-cols-2 gap-2.5">
                {FEATURE_ORDER.filter((k) => k in pred.features).map((k) => {
                  const m = pred.feature_meta[k] ?? {}
                  const v = pred.features[k]
                  const Icon = k === 'ghi' || k === 'cloud_cover' ? CloudSun : k === 'ndvi' ? Leaf : null
                  return (
                    <div key={k} className="rounded-xl bg-surface-2 px-3 py-2.5 ring-1 ring-line">
                      <dt className="flex items-center gap-1 text-[11px] text-muted">{Icon && <Icon size={12} weight="light" />}{m.label ?? k}</dt>
                      <dd className="mt-0.5 truncate text-sm">
                        {typeof v === 'number' ? <span className="font-mono tabular">{fmt(v, 2)}</span> : <span>{v ?? 'n/a'}</span>}
                        {typeof v === 'number' && m.unit ? <span className="ml-1 text-xs text-muted">{m.unit}</span> : null}
                      </dd>
                      <p className="mt-0.5 truncate text-[10px] text-muted">{m.source}</p>
                    </div>
                  )
                })}
              </dl>
            </Section>
          </>
        )}

        {!loading && !error && !pred && (
          <div className="p-6 text-sm text-muted">Click a location on land to analyse it.</div>
        )}
      </div>

      {pred && !loading && !error && (
        <div className="relative flex flex-wrap items-center gap-2 border-t border-line px-5 py-4">
          <PillButton href={api.reportUrl(pred.lat, pred.lon, params)} icon={<DownloadSimple size={16} weight="light" />}>
            Download report
          </PillButton>
          <PillButton variant="ghost" onClick={addCompare} icon={has(pred.lat, pred.lon) ? <CheckCircle size={16} weight="light" /> : <Plus size={16} weight="light" />}>
            Compare
          </PillButton>
          {toast && (
            <span role="status" className="absolute -top-10 left-5 rounded-full bg-ink px-3 py-1.5 text-xs text-bg shadow-[var(--shadow)]">{toast}</span>
          )}
        </div>
      )}
    </div>
  )
}
