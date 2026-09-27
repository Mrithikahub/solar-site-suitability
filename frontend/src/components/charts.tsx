import { animate, useMotionValue, useReducedMotion, useTransform, motion } from 'motion/react'
import { useEffect, useState } from 'react'
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { ShapItem } from '../lib/api'
import { fmt, signed } from '../lib/format'

/* Suitability ramp (single orange hue, light -> dark), validated sequential palette */
export const SUIT_STEPS = ['#fde7d9', '#f9c4a2', '#f39a6b', '#eb6834', '#c94f1f', '#9c3a14', '#6e280c']
/* Diverging SHAP poles: red raises the score, blue lowers it */
export const RAISE = '#e34948'
export const LOWER = '#2a78d6'
/* Categorical slots (fixed order) for up to three compared sites */
export const SERIES = ['#2a78d6', '#eb6834', '#1baf7a']

/* ------------------------------------------------------------------ Gauge */
export function ScoreGauge({ score, size = 176, label = 'Suitability' }: { score: number; size?: number; label?: string }) {
  const reduce = useReducedMotion()
  const mv = useMotionValue(0)
  const [display, setDisplay] = useState(0)
  useEffect(() => {
    const controls = animate(mv, score, { duration: reduce ? 0 : 1.1, ease: [0.16, 1, 0.3, 1] })
    const unsub = mv.on('change', (v) => setDisplay(v))
    return () => {
      controls.stop()
      unsub()
    }
  }, [score, reduce, mv])

  const stroke = 12
  const r = (size - stroke) / 2
  const c = size / 2
  const start = 135
  const sweep = 270
  const arc = (deg: number) => {
    const a = ((start + deg) * Math.PI) / 180
    return [c + r * Math.cos(a), c + r * Math.sin(a)]
  }
  const [x0, y0] = arc(0)
  const [x1, y1] = arc(sweep)
  const track = `M ${x0} ${y0} A ${r} ${r} 0 1 1 ${x1} ${y1}`
  const len = (sweep / 360) * 2 * Math.PI * r
  const dash = useTransform(mv, (v) => `${(Math.max(0, Math.min(100, v)) / 100) * len} ${len}`)
  const color = SUIT_STEPS[Math.min(SUIT_STEPS.length - 1, Math.floor((score / 100) * SUIT_STEPS.length))]

  return (
    <div className="relative" style={{ width: size, height: size }} role="img" aria-label={`${label} ${score.toFixed(0)} out of 100`}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <defs>
          <linearGradient id="gauge-grad" x1="0" y1="1" x2="1" y2="0">
            <stop offset="0%" stopColor={SUIT_STEPS[2]} />
            <stop offset="100%" stopColor={color} />
          </linearGradient>
        </defs>
        <path d={track} fill="none" stroke="var(--surface-3)" strokeWidth={stroke} strokeLinecap="round" />
        {/* hidden at 0: a zero-length arc would still draw its round end-cap as a dot */}
        <motion.path d={track} fill="none" stroke="url(#gauge-grad)" strokeWidth={stroke} strokeLinecap="round" style={{ strokeDasharray: dash, opacity: score < 0.5 ? 0 : 1 }} />
        {[40, 70].map((t) => {
          const a = ((start + (t / 100) * sweep) * Math.PI) / 180
          const ri = r - stroke / 2 - 3
          const ro = r + stroke / 2 + 3
          return <line key={t} x1={c + ri * Math.cos(a)} y1={c + ri * Math.sin(a)} x2={c + ro * Math.cos(a)} y2={c + ro * Math.sin(a)} stroke="var(--surface)" strokeWidth={2} />
        })}
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-mono text-[44px] leading-none font-medium tracking-tight tabular">{Math.round(display)}</span>
        <span className="mt-1.5 text-xs text-muted">{label} / 100</span>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ SHAP */
interface ShapRow { label: string; v: number; value: string }

function ShapTooltip({ active, payload }: { active?: boolean; payload?: { payload: ShapRow }[] }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="rounded-xl bg-surface px-3 py-2 text-xs shadow-[var(--shadow)] ring-1 ring-line">
      <p className="font-medium text-ink">{d.label}</p>
      <p className="text-ink-2">value: <span className="font-mono">{d.value}</span></p>
      <p className="text-ink-2">
        {d.v >= 0 ? 'raises' : 'lowers'} score by <span className="font-mono text-ink">{Math.abs(d.v).toFixed(1)}</span> pts
      </p>
    </div>
  )
}

export function ShapChart({ items, max = 8, height }: { items: ShapItem[]; max?: number; height?: number }) {
  // land-cover dummies: a contribution from value 0 means "is NOT this class"
  const nice = (i: ShapItem) => {
    if (i.feature.startsWith('lc_')) {
      const cls = i.label.replace('Land cover: ', '').toLowerCase()
      return i.value === 0 ? `Not ${cls}` : `Land cover: ${cls}`
    }
    return i.label.replace(' (log)', '')
  }
  const rows: ShapRow[] = items.slice(0, max).map((i) => ({
    label: nice(i),
    v: i.contribution_pct,
    value: i.value === null ? 'n/a' : `${fmt(i.value, 2)}${i.unit && !i.unit.startsWith('log') ? ` ${i.unit}` : ''}`,
  }))
  const lim = Math.max(4, ...rows.map((r) => Math.abs(r.v))) * 1.15
  return (
    <div style={{ height: height ?? rows.length * 30 + 24 }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} layout="vertical" margin={{ top: 0, right: 36, bottom: 0, left: 0 }} barCategoryGap={6}>
          <XAxis type="number" domain={[-lim, lim]} hide />
          <YAxis type="category" dataKey="label" width={138} tickLine={false} axisLine={false} tick={{ fill: 'var(--ink-2)', fontSize: 12 }} />
          <ReferenceLine x={0} stroke="var(--line-strong)" />
          <Tooltip cursor={{ fill: 'var(--surface-2)' }} content={<ShapTooltip />} />
          <Bar dataKey="v" radius={4} isAnimationActive label={{ position: 'right', formatter: (v: unknown) => signed(Number(v)), fill: 'var(--muted)', fontSize: 11 }}>
            {rows.map((r) => (
              <Cell key={r.label} fill={r.v >= 0 ? RAISE : LOWER} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

/* ------------------------------------------------------------------ Legend ramp */
export function RampLegend({ palette, min, max, unit, title }: { palette: string[]; min: number; max: number; unit?: string; title: string }) {
  return (
    <div className="space-y-1.5">
      <p className="text-xs font-medium text-ink">{title}</p>
      <div className="h-2 w-full rounded-full ring-1 ring-line" style={{ background: `linear-gradient(90deg, ${palette.join(',')})` }} />
      <div className="flex justify-between font-mono text-[10.5px] text-muted tabular">
        <span>{fmt(min, 2)}</span>
        {unit ? <span>{unit}</span> : null}
        <span>{fmt(max, 2)}</span>
      </div>
    </div>
  )
}
