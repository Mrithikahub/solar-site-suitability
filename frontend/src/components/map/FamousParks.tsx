import { SolarPanel } from '@phosphor-icons/react'
import { useEffect, useState } from 'react'
import { api, type GeocodeResult } from '../../lib/api'

/** Quick-pick chips for well-known solar parks (handy for demos). */
export function FamousParks({ onPick }: { onPick: (r: GeocodeResult) => void }) {
  const [places, setPlaces] = useState<GeocodeResult[]>([])
  useEffect(() => {
    let cancelled = false
    api.famousPlaces().then((r) => !cancelled && setPlaces(r.places)).catch(() => setPlaces([]))
    return () => {
      cancelled = true
    }
  }, [])
  if (!places.length) return null

  return (
    <div className="glass pointer-events-auto flex max-w-full items-center gap-2 rounded-full bg-[var(--glass)] py-1.5 pl-4 pr-1.5 shadow-[var(--shadow)] ring-1 ring-line backdrop-blur-xl">
      <span className="flex shrink-0 items-center gap-1.5 text-xs text-ink-2">
        <SolarPanel size={16} weight="light" className="text-accent" />
        <span className="hidden sm:inline">Try a famous solar park</span>
      </span>
      <div className="flex gap-1.5 overflow-x-auto [scrollbar-width:none]">
        {places.map((p) => (
          <button
            key={p.name}
            type="button"
            onClick={() => onPick(p)}
            title={`${p.name}: ${p.display_name}`}
            className="shrink-0 rounded-full bg-ink/[0.05] px-3 py-1.5 text-xs text-ink ring-1 ring-line transition-all duration-300 ease-[cubic-bezier(0.32,0.72,0,1)] hover:bg-accent-soft hover:text-accent-ink active:scale-95 dark:bg-white/[0.05]"
          >
            {p.name.replace(/( Al Maktoum)? (Solar Power Project|Solar Park|Solar Farm|Ultra Mega Solar Park|Ultra Mega Solar|Solar Complex|Desert Solar Park)$/, '')}
          </button>
        ))}
      </div>
    </div>
  )
}
