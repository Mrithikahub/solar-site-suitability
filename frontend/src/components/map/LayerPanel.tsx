import { CaretDown, Eye, EyeSlash, StackSimple } from '@phosphor-icons/react'
import { AnimatePresence, motion } from 'motion/react'
import { useState } from 'react'
import type { GeocodeResult, LayerEntry, Metric } from '../../lib/api'
import type { BasemapKind } from '../../lib/basemaps'
import { RampLegend } from '../charts'
import { SearchBox } from '../SearchBox'
import { cx, EASE, Segmented } from '../ui'
import { CONTEXT_LAYERS, type LayerCatalog } from './layers'

const METRIC_HELP: Record<Metric, string> = {
  suitability: 'Physical suitability: sun, cloud, slope, vegetation, land cover.',
  development: 'Where plants have historically been built: near infrastructure and demand.',
  ahp: 'Expert-weighted GIS overlay with exclusions (baseline).',
}

function Legend({ layer }: { layer: LayerEntry }) {
  if (layer.kind === 'categorical' && layer.classes) {
    return (
      <div>
        <p className="mb-2 text-xs font-medium">{layer.title}</p>
        <ul className="grid grid-cols-2 gap-x-3 gap-y-1.5">
          {layer.classes.map((c) => (
            <li key={c.code} className="flex items-center gap-2 text-[11px] text-ink-2">
              <span className="h-2.5 w-2.5 shrink-0 rounded-[3px] ring-1 ring-line" style={{ background: c.color }} />
              <span className="truncate">{c.name}</span>
            </li>
          ))}
        </ul>
      </div>
    )
  }
  if (layer.kind === 'image') return <p className="text-xs text-ink-2">{layer.title}: natural colour (B4, B3, B2).</p>
  return <RampLegend palette={layer.palette ?? []} min={layer.min ?? 0} max={layer.max ?? 1} unit={layer.unit} title={layer.title} />
}

export function LayerPanel({
  catalog, metric, setMetric, showScore, setShowScore, context, setContext, opacity, setOpacity, tn, setTn, baseKind, setBaseKind, onPlace,
}: {
  catalog: LayerCatalog | null
  metric: Metric
  setMetric: (m: Metric) => void
  showScore: boolean
  setShowScore: (v: boolean) => void
  context: string | null
  setContext: (k: string | null) => void
  opacity: number
  setOpacity: (v: number) => void
  tn: boolean
  setTn: (v: boolean) => void
  baseKind: BasemapKind
  setBaseKind: (b: BasemapKind) => void
  onPlace: (r: GeocodeResult) => void
}) {
  const [open, setOpen] = useState(() => window.matchMedia('(min-width: 768px)').matches)
  const scope = tn ? 'tamil_nadu' : 'global'
  const scoreLayer = catalog?.[scope]?.[metric] ?? catalog?.global?.[metric]
  const ctxLayer = context ? catalog?.[scope]?.[context] : undefined
  const available = CONTEXT_LAYERS.filter((l) => catalog?.[scope]?.[l.key])

  return (
    <div className="pointer-events-auto flex w-full flex-col gap-3 md:w-[340px]">
      <SearchBox onSelect={onPlace} />
      <div className="glass overflow-hidden rounded-[1.6rem] bg-[var(--glass)] shadow-[var(--shadow)] ring-1 ring-line backdrop-blur-xl">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="flex w-full items-center justify-between px-5 py-3.5 text-left"
          aria-expanded={open}
        >
          <span className="flex items-center gap-2 text-sm font-medium"><StackSimple size={18} weight="light" className="text-accent" />Layers</span>
          <CaretDown size={16} weight="light" className={cx('text-muted transition-transform duration-500 ease-[cubic-bezier(0.32,0.72,0,1)]', open && 'rotate-180')} />
        </button>
        <AnimatePresence initial={false}>
          {open && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.45, ease: EASE }}
            >
              <div className="max-h-[calc(100dvh-260px)] space-y-5 overflow-y-auto px-5 pb-5">
                <div className="space-y-2.5">
                  <div className="flex items-center justify-between">
                    <p className="text-xs font-medium text-ink-2">Score overlay</p>
                    <button type="button" onClick={() => setShowScore(!showScore)} className="flex items-center gap-1 text-xs text-muted hover:text-ink" aria-pressed={showScore}>
                      {showScore ? <Eye size={14} weight="light" /> : <EyeSlash size={14} weight="light" />}
                      {showScore ? 'Visible' : 'Hidden'}
                    </button>
                  </div>
                  <Segmented
                    label="Score overlay"
                    size="sm"
                    value={metric}
                    onChange={setMetric}
                    options={[
                      { value: 'suitability', label: 'Suitability' },
                      { value: 'development', label: 'Development' },
                      { value: 'ahp', label: 'AHP' },
                    ]}
                  />
                  <p className="text-[11.5px] leading-snug text-muted">{METRIC_HELP[metric]}</p>
                  {tn && metric === 'development' && (
                    <p className="text-[11.5px] leading-snug text-warn">Development likelihood has no Tamil Nadu high-res layer; the global grid is shown.</p>
                  )}
                </div>

                <div className="space-y-2.5">
                  <p className="text-xs font-medium text-ink-2">Remote-sensing layer</p>
                  <div className="flex flex-wrap gap-1.5">
                    <Chip active={context === null} onClick={() => setContext(null)}>None</Chip>
                    {available.map((l) => (
                      <Chip key={l.key} active={context === l.key} onClick={() => setContext(l.key)}>{l.label}</Chip>
                    ))}
                  </div>
                  <label className="flex items-center gap-3 pt-1 text-xs text-ink-2">
                    <span className="w-14 shrink-0">Opacity</span>
                    <input type="range" min={0.2} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} className="w-full" aria-label="Overlay opacity" />
                  </label>
                </div>

                <div className="space-y-2.5">
                  <p className="text-xs font-medium text-ink-2">Basemap</p>
                  <Segmented label="Basemap" size="sm" value={baseKind} onChange={setBaseKind}
                    options={[{ value: 'map', label: 'Map' }, { value: 'satellite', label: 'Satellite imagery' }]} />
                </div>

                <label className="flex cursor-pointer items-center justify-between gap-3 rounded-2xl bg-ink/[0.04] px-3.5 py-3 ring-1 ring-line dark:bg-white/[0.04]">
                  <span>
                    <span className="block text-[13px] font-medium">Tamil Nadu at 285 m</span>
                    <span className="block text-[11.5px] text-muted">Sentinel-2, Landsat, SRTM case-study layers</span>
                  </span>
                  <span className="relative inline-flex h-6 w-10 shrink-0 items-center">
                    <input type="checkbox" className="peer sr-only" checked={tn} onChange={(e) => setTn(e.target.checked)} />
                    <span className="absolute inset-0 rounded-full bg-ink/20 ring-1 ring-line-strong transition-colors duration-300 peer-checked:bg-accent dark:bg-white/20" />
                    <span className="absolute left-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] peer-checked:translate-x-4" />
                  </span>
                </label>

                <div className="space-y-4 border-t border-line pt-4">
                  {showScore && scoreLayer && <Legend layer={scoreLayer} />}
                  {ctxLayer && <Legend layer={ctxLayer} />}
                  {ctxLayer && <p className="text-[10.5px] text-muted">Source: {ctxLayer.source}</p>}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

function Chip({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cx(
        'rounded-full px-3 py-1 text-xs transition-all duration-300 ease-[cubic-bezier(0.32,0.72,0,1)] active:scale-95',
        active ? 'bg-ink text-bg' : 'bg-ink/[0.05] text-ink-2 ring-1 ring-line hover:text-ink dark:bg-white/[0.05]',
      )}
    >
      {children}
    </button>
  )
}
