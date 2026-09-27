import { useCallback, useEffect, useState } from 'react'

export interface SavedSite {
  id: string
  name: string
  lat: number
  lon: number
}

const KEY = 'solarsite-compare'
const EVT = 'solarsite-compare-change'
export const MAX_COMPARE = 3

function read(): SavedSite[] {
  try {
    const raw = localStorage.getItem(KEY)
    const v = raw ? (JSON.parse(raw) as SavedSite[]) : []
    return Array.isArray(v) ? v.slice(0, MAX_COMPARE) : []
  } catch {
    return []
  }
}

function write(sites: SavedSite[]) {
  try {
    localStorage.setItem(KEY, JSON.stringify(sites.slice(0, MAX_COMPARE)))
  } catch {
    /* storage unavailable: list lives only for this page view */
  }
  window.dispatchEvent(new Event(EVT))
}

export function siteId(lat: number, lon: number) {
  return `${lat.toFixed(4)},${lon.toFixed(4)}`
}

/** Sites queued for the compare view (per viewer convenience, localStorage). */
export function useCompareSites() {
  const [sites, setSites] = useState<SavedSite[]>(read)

  useEffect(() => {
    const sync = () => setSites(read())
    window.addEventListener(EVT, sync)
    window.addEventListener('storage', sync)
    return () => {
      window.removeEventListener(EVT, sync)
      window.removeEventListener('storage', sync)
    }
  }, [])

  const add = useCallback((s: Omit<SavedSite, 'id'>) => {
    const cur = read()
    const id = siteId(s.lat, s.lon)
    if (cur.some((c) => c.id === id)) return 'exists' as const
    if (cur.length >= MAX_COMPARE) return 'full' as const
    write([...cur, { ...s, id }])
    return 'added' as const
  }, [])

  const remove = useCallback((id: string) => write(read().filter((s) => s.id !== id)), [])
  const clear = useCallback(() => write([]), [])
  const has = useCallback((lat: number, lon: number) => sites.some((s) => s.id === siteId(lat, lon)), [sites])

  return { sites, add, remove, clear, has }
}
