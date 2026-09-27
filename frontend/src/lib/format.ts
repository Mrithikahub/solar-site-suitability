export const ACRE_M2 = 4046.8564224

export function fmt(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return 'n/a'
  const abs = Math.abs(v)
  if (abs >= 10000) return Math.round(v).toLocaleString('en-US')
  if (abs >= 1000) return v.toLocaleString('en-US', { maximumFractionDigits: 0 })
  return v.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: digits })
}

export function coord(lat: number, lon: number): string {
  const ns = lat >= 0 ? 'N' : 'S'
  const ew = lon >= 0 ? 'E' : 'W'
  return `${Math.abs(lat).toFixed(4)}° ${ns}, ${Math.abs(lon).toFixed(4)}° ${ew}`
}

export function signed(v: number, digits = 1): string {
  const s = v.toFixed(digits)
  return v > 0 ? `+${s}` : s
}

export function classFor(score: number): 'High' | 'Medium' | 'Low' {
  return score >= 70 ? 'High' : score >= 40 ? 'Medium' : 'Low'
}
