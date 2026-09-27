import { useEffect, useState } from 'react'
import { api, API_URL, staticJson, type LayerEntry, type Scope } from '../../lib/api'

export type LayerCatalog = Record<Scope, Record<string, LayerEntry>>

type ManifestEntry = Omit<LayerEntry, 'name' | 'scope' | 'image_url'>

/** Layer catalogue from the API, falling back to the manifest shipped with the site. */
export function useLayerCatalog() {
  const [catalog, setCatalog] = useState<LayerCatalog | null>(null)
  const [offline, setOffline] = useState(false)

  useEffect(() => {
    let cancelled = false
    const toMap = (layers: LayerEntry[]) => Object.fromEntries(layers.map((l) => [l.name, l]))
    Promise.all([api.layers('global'), api.layers('tamil_nadu')])
      .then(([g, t]) => !cancelled && setCatalog({ global: toMap(g.layers), tamil_nadu: toMap(t.layers) }))
      .catch(async () => {
        if (cancelled) return
        setOffline(true)
        try {
          const m = await staticJson<Record<Scope, Record<string, ManifestEntry>>>('layers_manifest.json')
          const build = (scope: Scope) =>
            Object.fromEntries(
              Object.entries(m[scope]).map(([name, e]) => [
                name,
                { ...e, name, scope, image_url: `${API_URL}/layers/${name}/image?scope=${scope}` } as LayerEntry,
              ]),
            )
          setCatalog({ global: build('global'), tamil_nadu: build('tamil_nadu') })
        } catch {
          setCatalog({ global: {}, tamil_nadu: {} })
        }
      })
    return () => {
      cancelled = true
    }
  }, [])

  return { catalog, offline }
}

export const CONTEXT_LAYERS: { key: string; label: string; tnOnly?: boolean }[] = [
  { key: 'ghi', label: 'Solar irradiance' },
  { key: 'ndvi', label: 'NDVI' },
  { key: 'slope', label: 'Slope' },
  { key: 'landcover', label: 'Land cover' },
  { key: 'lst', label: 'Surface temperature' },
  { key: 'nightlights', label: 'Night lights' },
  { key: 'population', label: 'Population' },
  { key: 'ndbi', label: 'NDBI', tnOnly: true },
  { key: 'elevation', label: 'Elevation', tnOnly: true },
  { key: 'truecolor', label: 'Sentinel-2 true colour', tnOnly: true },
]
