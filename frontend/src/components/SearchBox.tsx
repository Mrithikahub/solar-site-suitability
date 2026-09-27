import { CircleNotch, Crosshair, GlobeHemisphereEast, MagnifyingGlass, MapPin } from '@phosphor-icons/react'
import { AnimatePresence, motion } from 'motion/react'
import { useCallback, useEffect, useId, useRef, useState } from 'react'
import { api, parseCoordinates, type GeocodeResult } from '../lib/api'
import { cx, EASE } from './ui'

const DEBOUNCE_MS = 300

function KindIcon({ r }: { r: GeocodeResult }) {
  if (r.type === 'coordinates') return <Crosshair size={16} weight="light" className="mt-0.5 shrink-0 text-accent" />
  if (r.kind === 'area') return <GlobeHemisphereEast size={16} weight="light" className="mt-0.5 shrink-0 text-ink-2" />
  return <MapPin size={16} weight="light" className="mt-0.5 shrink-0 text-accent" />
}

/**
 * Place search with autocomplete (Photon + Nominatim via the backend).
 * Arrow keys move through the top 5 suggestions; Enter selects the highlighted
 * one, or the top result when nothing is highlighted. Pasted coordinates
 * ("51.5294, -0.1727") resolve instantly without a network call.
 */
export function SearchBox({ onSelect, placeholder = 'Search a place, landmark or coordinates', className, autoFocus }: {
  onSelect: (r: GeocodeResult) => void
  placeholder?: string
  className?: string
  autoFocus?: boolean
}) {
  const [q, setQ] = useState('')
  const [results, setResults] = useState<GeocodeResult[]>([])
  const [resultsFor, setResultsFor] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(-1)
  const listId = useId()
  const box = useRef<HTMLDivElement>(null)
  const inflight = useRef<AbortController | null>(null)
  const pickOnArrival = useRef(false)

  const pick = useCallback((r: GeocodeResult) => {
    onSelect(r)
    inflight.current?.abort()
    setQ(r.name)
    setResultsFor(r.name.trim())   // the box now shows a resolved place: don't search it again
    setLoading(false)
    setOpen(false)
    setActive(-1)
    pickOnArrival.current = false
  }, [onSelect])

  const fetchNow = useCallback(async (query: string) => {
    inflight.current?.abort()
    const ctrl = new AbortController()
    inflight.current = ctrl
    setLoading(true)
    setError(null)
    try {
      const r = await api.geocode(query, ctrl.signal)
      if (ctrl.signal.aborted) return
      setResults(r.results)
      setResultsFor(query)
      setActive(-1)
      setOpen(true)
      if (pickOnArrival.current && r.results.length) pick(r.results[0])
    } catch (e) {
      if (!ctrl.signal.aborted) setError((e as Error).message)
    } finally {
      if (!ctrl.signal.aborted) setLoading(false)
    }
  }, [pick])

  // autocomplete (debounced); coordinates resolve locally
  useEffect(() => {
    const query = q.trim()
    const coords = parseCoordinates(query)
    if (coords) {
      inflight.current?.abort()
      setResults([coords])
      setResultsFor(query)
      setLoading(false)
      setError(null)
      return
    }
    if (query.length < 2) {
      setResults([])
      setResultsFor('')
      setError(null)
      return
    }
    if (query === resultsFor) return
    const t = window.setTimeout(() => fetchNow(query), DEBOUNCE_MS)
    return () => window.clearTimeout(t)
  }, [q, resultsFor, fetchNow])

  useEffect(() => () => inflight.current?.abort(), [])

  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [])

  const onEnter = () => {
    const query = q.trim()
    if (!query) return
    const coords = parseCoordinates(query)
    if (coords) return pick(coords)
    if (resultsFor === query && results.length) return pick(results[active >= 0 ? active : 0])
    // results for the current text are not in yet: fetch now and take the top hit
    pickOnArrival.current = true
    fetchNow(query)
  }

  const showList = open && (results.length > 0 || error || (resultsFor === q.trim() && !loading && q.trim().length >= 2))

  return (
    <div ref={box} className={cx('relative', className)}>
      <label htmlFor={`${listId}-input`} className="sr-only">Search places</label>
      <div className="glass flex h-12 items-center gap-2 rounded-full bg-[var(--glass)] pl-4 pr-3 shadow-[var(--shadow)] ring-1 ring-line backdrop-blur-xl focus-within:ring-accent/60">
        <MagnifyingGlass size={18} weight="light" className="shrink-0 text-muted" />
        <input
          id={`${listId}-input`}
          role="combobox"
          aria-expanded={!!showList}
          aria-controls={listId}
          aria-autocomplete="list"
          aria-activedescendant={active >= 0 ? `${listId}-opt-${active}` : undefined}
          autoFocus={autoFocus}
          autoComplete="off"
          spellCheck={false}
          value={q}
          onChange={(e) => { setQ(e.target.value); setOpen(true) }}
          onFocus={() => results.length && setOpen(true)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') { e.preventDefault(); onEnter(); return }
            if (e.key === 'Escape') { setOpen(false); setActive(-1); return }
            if (!results.length) return
            if (e.key === 'ArrowDown') { e.preventDefault(); setOpen(true); setActive((a) => (a + 1) % results.length) }
            if (e.key === 'ArrowUp') { e.preventDefault(); setOpen(true); setActive((a) => (a <= 0 ? results.length - 1 : a - 1)) }
          }}
          placeholder={placeholder}
          className="h-full min-w-0 flex-1 bg-transparent text-[14px] text-ink placeholder:text-muted focus:outline-none"
        />
        {loading && <CircleNotch size={16} className="animate-spin text-muted" aria-label="Searching" />}
      </div>
      <AnimatePresence>
        {showList && (
          <motion.ul
            id={listId}
            role="listbox"
            initial={{ opacity: 0, y: -6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.98 }}
            transition={{ duration: 0.25, ease: EASE }}
            className="absolute inset-x-0 top-14 z-50 overflow-hidden rounded-2xl bg-surface p-1.5 shadow-[var(--shadow)] ring-1 ring-line"
          >
            {error ? (
              <li className="px-3 py-2.5 text-sm text-bad">{error}</li>
            ) : results.length === 0 ? (
              <li className="px-3 py-2.5 text-sm text-muted">No places found. Try a nearby town, or paste coordinates like 51.5294, -0.1727.</li>
            ) : (
              results.map((r, i) => (
                <li key={`${r.lat},${r.lon},${i}`} id={`${listId}-opt-${i}`} role="option" aria-selected={i === active}>
                  <button
                    type="button"
                    onMouseEnter={() => setActive(i)}
                    onClick={() => pick(r)}
                    className={cx('flex w-full items-start gap-3 rounded-xl px-3 py-2.5 text-left transition-colors duration-200', i === active ? 'bg-surface-2' : '')}
                  >
                    <KindIcon r={r} />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm font-medium text-ink">{r.name}</span>
                      <span className="block truncate text-xs text-muted">{r.display_name}</span>
                    </span>
                    <span className="mt-0.5 shrink-0 rounded-full bg-ink/[0.06] px-2 py-0.5 text-[10.5px] capitalize text-ink-2 ring-1 ring-line dark:bg-white/[0.06]">
                      {r.type}
                    </span>
                  </button>
                </li>
              ))
            )}
            {results.length > 0 && !error && (
              <li className="px-3 pb-1 pt-1.5 text-[10.5px] text-muted" aria-hidden>
                {results[0].kind === 'point' || results[0].type === 'coordinates'
                  ? 'Enter jumps to the top result and analyses it'
                  : 'Enter jumps to the top result'}
              </li>
            )}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  )
}
