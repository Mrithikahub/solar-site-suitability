import { ArrowUpRight, CheckCircle, MinusCircle, XCircle } from '@phosphor-icons/react'
import { motion, useReducedMotion } from 'motion/react'
import type { ComponentProps, ReactNode } from 'react'
import { Link } from 'react-router-dom'
import type { ScoreClass } from '../lib/api'

export const EASE = [0.32, 0.72, 0, 1] as const
export const cx = (...c: (string | false | null | undefined)[]) => c.filter(Boolean).join(' ')

/* ------------------------------------------------------------------ Bezel
 * "Double-bezel" container: an outer tray (hairline ring, tinted, padded)
 * holding an inner core with its own highlight and a concentric radius. */
export function Bezel({ children, className, inner, radius = '2rem', pad = '0.375rem' }: {
  children: ReactNode
  className?: string
  inner?: string
  radius?: string
  pad?: string
}) {
  return (
    <div
      className={cx('bg-ink/[0.035] ring-1 ring-line dark:bg-white/[0.035]', className)}
      style={{ borderRadius: radius, padding: pad }}
    >
      <div
        className={cx('h-full overflow-hidden bg-surface shadow-[inset_0_1px_1px_rgba(255,255,255,0.6)] dark:shadow-[inset_0_1px_1px_rgba(255,255,255,0.06)]', inner)}
        style={{ borderRadius: `calc(${radius} - ${pad})` }}
      >
        {children}
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ Buttons */
type PillProps = {
  children: ReactNode
  variant?: 'primary' | 'ghost'
  icon?: ReactNode
  className?: string
} & ({ to: string; href?: never; onClick?: never } | { href: string; to?: never; onClick?: never; download?: boolean } | { onClick: () => void; to?: never; href?: never; disabled?: boolean })

export function PillButton(props: PillProps) {
  const { children, variant = 'primary', icon, className } = props
  const base = cx(
    'group inline-flex items-center gap-3 rounded-full py-2 pl-5 pr-2 text-[15px] font-medium whitespace-nowrap select-none',
    'transition-[transform,background-color,box-shadow] duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] active:scale-[0.98]',
    variant === 'primary'
      ? 'bg-ink text-bg shadow-[0_10px_30px_-10px_rgba(0,0,0,0.45)] hover:shadow-[0_14px_36px_-12px_rgba(0,0,0,0.55)]'
      : 'bg-ink/[0.05] text-ink ring-1 ring-line hover:bg-ink/[0.08] dark:bg-white/[0.05] dark:hover:bg-white/[0.09]',
    'disabled:opacity-50 disabled:pointer-events-none',
    className,
  )
  const nub = (
    <span
      className={cx(
        'flex h-8 w-8 items-center justify-center rounded-full transition-transform duration-500 ease-[cubic-bezier(0.32,0.72,0,1)]',
        'group-hover:translate-x-0.5 group-hover:-translate-y-px group-hover:scale-105',
        variant === 'primary' ? 'bg-bg/15' : 'bg-ink/[0.07] dark:bg-white/10',
      )}
    >
      {icon ?? <ArrowUpRight size={16} weight="light" />}
    </span>
  )
  if ('to' in props && props.to) return <Link to={props.to} className={base}>{children}{nub}</Link>
  if ('href' in props && props.href)
    return <a href={props.href} className={base} download={'download' in props ? props.download : undefined}>{children}{nub}</a>
  const p = props as { onClick: () => void; disabled?: boolean }
  return <button type="button" onClick={p.onClick} disabled={p.disabled} className={base}>{children}{nub}</button>
}

export function IconButton({ label, className, children, ...rest }: ComponentProps<'button'> & { label: string }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className={cx(
        'flex h-9 w-9 items-center justify-center rounded-full text-ink-2 transition duration-300 ease-[cubic-bezier(0.32,0.72,0,1)]',
        'hover:bg-ink/[0.06] hover:text-ink active:scale-95 dark:hover:bg-white/[0.08]',
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  )
}

/* ------------------------------------------------------------------ Reveal */
export function Reveal({ children, delay = 0, className, y = 28 }: { children: ReactNode; delay?: number; className?: string; y?: number }) {
  const reduce = useReducedMotion()
  return (
    <motion.div
      className={className}
      initial={reduce ? false : { opacity: 0, y, filter: 'blur(8px)' }}
      whileInView={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
      viewport={{ once: true, amount: 0.25 }}
      transition={{ duration: 0.9, delay, ease: EASE }}
    >
      {children}
    </motion.div>
  )
}

/* ------------------------------------------------------------------ Status */
const CLASS_META: Record<ScoreClass, { color: string; Icon: typeof CheckCircle }> = {
  High: { color: 'text-good', Icon: CheckCircle },
  Medium: { color: 'text-warn', Icon: MinusCircle },
  Low: { color: 'text-bad', Icon: XCircle },
}

export function ClassBadge({ cls, size = 'md' }: { cls: ScoreClass; size?: 'sm' | 'md' }) {
  const { color, Icon } = CLASS_META[cls]
  return (
    <span className={cx('inline-flex items-center gap-1 font-medium', color, size === 'sm' ? 'text-xs' : 'text-sm')}>
      <Icon size={size === 'sm' ? 14 : 16} weight="fill" />
      <span>{cls}</span>
    </span>
  )
}

/* ------------------------------------------------------------------ Segmented control */
export function Segmented<T extends string>({ value, options, onChange, size = 'md', label }: {
  value: T
  options: { value: T; label: string }[]
  onChange: (v: T) => void
  size?: 'sm' | 'md'
  label: string
}) {
  return (
    <div role="radiogroup" aria-label={label} className="relative flex rounded-full bg-ink/[0.05] p-1 ring-1 ring-line dark:bg-white/[0.05]">
      {options.map((o) => {
        const active = o.value === value
        return (
          <button
            key={o.value}
            role="radio"
            aria-checked={active}
            type="button"
            onClick={() => onChange(o.value)}
            className={cx(
              'relative z-10 flex-1 rounded-full font-medium whitespace-nowrap transition-colors duration-300',
              size === 'sm' ? 'px-3 py-1 text-xs' : 'px-4 py-1.5 text-[13px]',
              active ? 'text-ink' : 'text-muted hover:text-ink-2',
            )}
          >
            {active && (
              <motion.span
                layoutId={`seg-${label}`}
                className="absolute inset-0 -z-10 rounded-full bg-surface shadow-[0_1px_3px_rgba(0,0,0,0.12)] ring-1 ring-line"
                transition={{ type: 'spring', stiffness: 380, damping: 32 }}
              />
            )}
            {o.label}
          </button>
        )
      })}
    </div>
  )
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cx('skeleton', className)} aria-hidden />
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="rounded-md bg-ink/[0.06] px-1.5 py-0.5 font-mono text-[11px] text-ink-2 ring-1 ring-line dark:bg-white/[0.06]">{children}</kbd>
}
