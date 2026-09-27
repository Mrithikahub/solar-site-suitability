import { CursorClick, WarningCircle } from '@phosphor-icons/react'
import L from 'leaflet'
import { AnimatePresence, motion } from 'motion/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ImageOverlay, MapContainer, Marker, Pane, TileLayer, useMapEvents } from 'react-leaflet'
import { useSearchParams } from 'react-router-dom'
import { FamousParks } from '../components/map/FamousParks'
import { LayerPanel } from '../components/map/LayerPanel'
import { useLayerCatalog } from '../components/map/layers'
import { ResultPanel, type EnergyInputs } from '../components/map/ResultPanel'
import { EASE } from '../components/ui'
import { api, ApiError, type GeocodeResult, type Metric, type Prediction } from '../lib/api'
import { basemap, ESRI_ATTRIBUTION, type BasemapKind } from '../lib/basemaps'
import { ACRE_M2 } from '../lib/format'
import { useTheme } from '../lib/theme'

const markerIcon = L.divIcon({ className: '', html: '<div class="pulse-marker"><span></span></div>', iconSize: [18, 18], iconAnchor: [9, 9] })

function ClickCatcher({ onClick, onZoom }: { onClick: (lat: number, lon: number) => void; onZoom: (z: number) => void }) {
  const map = useMapEvents({
    click(e) {
      const { lat, lng } = e.latlng.wrap()
      onClick(lat, lng)
    },
    zoomend() {
      onZoom(map.getZoom())
    },
  })
  return null
}

const TN_BOUNDS: L.LatLngBoundsExpression = [[8.0, 76.2], [13.6, 80.4]]

export default function MapPage() {
  const { theme } = useTheme()
  const { catalog, offline } = useLayerCatalog()
  const [params, setParams] = useSearchParams()
  const [map, setMap] = useState<L.Map | null>(null)

  const [metric, setMetric] = useState<Metric>('suitability')
  const [showScore, setShowScore] = useState(true)
  const [context, setContext] = useState<string | null>(null)
  const [opacity, setOpacity] = useState(0.85)
  const [tn, setTnState] = useState(false)
  const [baseKind, setBaseKind] = useState<BasemapKind>('map')
  // basemap to restore after an automatic switch to satellite for a searched place
  const autoBasePrev = useRef<BasemapKind | null>(null)
  const chooseBase = (b: BasemapKind) => {
    autoBasePrev.current = null          // a manual choice always wins
    setBaseKind(b)
  }
  const [zoom, setZoom] = useState(3)

  const [point, setPoint] = useState<{ lat: number; lon: number } | null>(null)
  const [pred, setPred] = useState<Prediction | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [inputs, setInputs] = useState<EnergyInputs>({ acres: 1, efficiency: 20, pr: 75 })
  const abortRef = useRef<AbortController | null>(null)

  const scope = tn ? 'tamil_nadu' : 'global'
  const scoreLayer = catalog?.[scope]?.[metric] ?? catalog?.global?.[metric]
  const ctxLayer = context ? catalog?.[scope]?.[context] : undefined

  const setTn = (v: boolean) => {
    setTnState(v)
    if (v) map?.flyToBounds(TN_BOUNDS, { duration: 1.4 })
    if (context && !catalog?.[v ? 'tamil_nadu' : 'global']?.[context]) setContext(null)
  }

  const analyse = useCallback(async (lat: number, lon: number, e: EnergyInputs, soft = false) => {
    abortRef.current?.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl
    if (!soft) {
      setLoading(true)
      setPred(null)
    }
    setError(null)
    try {
      const r = await api.predict(lat, lon, { area_m2: e.acres * ACRE_M2, efficiency: e.efficiency / 100, performance_ratio: e.pr / 100 }, ctrl.signal)
      if (!ctrl.signal.aborted) setPred(r)
    } catch (err) {
      if (ctrl.signal.aborted) return
      const ae = err as ApiError
      setError(ae.status === 404 ? 'This point is open water or outside the land the model covers. Try a location on land.' : ae.message)
    } finally {
      if (!ctrl.signal.aborted) setLoading(false)
    }
  }, [])

  /** Analyse a point. `flyZoom` flies there first (search results); otherwise the map pans gently (clicks). */
  const select = useCallback((lat: number, lon: number, flyZoom?: number) => {
    setPoint({ lat, lon })
    if (map) {
      // keep the marker visible beside the result panel (right sheet) or above it (mobile bottom sheet)
      const z = flyZoom ?? map.getZoom()
      const size = map.getSize()
      const mobile = size.x < 768
      const offset = mobile ? L.point(0, size.y * 0.3) : L.point(Math.min(240, size.x * 0.15), 0)
      const target = map.unproject(map.project([lat, lon], z).add(offset), z)
      if (flyZoom !== undefined) map.flyTo(target, z, { duration: 1.6 })
      else map.panTo(target, { animate: true, duration: 0.8 })
    }
    setParams({ lat: lat.toFixed(4), lon: lon.toFixed(4) }, { replace: true })
    analyse(lat, lon, inputs)
  }, [analyse, inputs, setParams, map])

  // dev-only handle for automated end-to-end checks (stripped from production builds)
  useEffect(() => {
    if (import.meta.env.DEV) (window as unknown as { __solarMap?: L.Map | null }).__solarMap = map
  }, [map])

  // deep link: /map?lat=..&lon=..
  useEffect(() => {
    const lat = Number(params.get('lat'))
    const lon = Number(params.get('lon'))
    if (map && params.get('lat') && !Number.isNaN(lat) && !Number.isNaN(lon) && !point) {
      map.setView([lat, lon], 5)
      select(lat, lon)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map])

  // energy inputs changed -> re-run (features are cached server-side, so this is quick)
  const firstInputs = useRef(true)
  useEffect(() => {
    if (firstInputs.current) { firstInputs.current = false; return }
    if (!point) return
    const t = window.setTimeout(() => analyse(point.lat, point.lon, inputs, true), 450)
    return () => window.clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inputs])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  const close = () => {
    abortRef.current?.abort()
    setPoint(null)
    setPred(null)
    setError(null)
    setLoading(false)
    setParams({}, { replace: true })
  }

  /**
   * Search result: a specific place (POI, building, address, coordinates) is
   * treated like a click - fly to the exact point, drop the marker, analyse.
   * A city / region / country fits its bounding box without analysis.
   */
  const onPlace = (r: GeocodeResult) => {
    if (!map) return
    if (r.kind === 'point') {
      // a specific site is easiest to judge on imagery: switch to satellite, remember the previous basemap
      if (baseKind !== 'satellite') {
        autoBasePrev.current = baseKind
        setBaseKind('satellite')
      }
      select(r.lat, r.lon, Math.min(r.zoom ?? 16, 17))
      return
    }
    if (autoBasePrev.current) {
      setBaseKind(autoBasePrev.current)
      autoBasePrev.current = null
    }
    if (r.bbox) {
      const [s, n, w, e] = r.bbox
      map.flyToBounds([[s, w], [n, e]], { duration: 1.4, padding: [40, 40] })
    } else map.flyTo([r.lat, r.lon], r.zoom ?? 10, { duration: 1.4 })
  }

  const tiles = useMemo(() => basemap(baseKind, theme), [baseKind, theme])

  const toBounds = (b: { south: number; west: number; north: number; east: number }): L.LatLngBoundsExpression => [[b.south, b.west], [b.north, b.east]]
  const panelOpen = point !== null

  return (
    <div className="fixed inset-0">
      <MapContainer
        ref={setMap}
        center={[22, 20]}
        zoom={3}
        minZoom={2}
        maxZoom={18}
        worldCopyJump
        zoomControl={false}
        className="h-full w-full"
        maxBounds={[[-85, -240], [85, 240]]}
        attributionControl
      >
        <TileLayer key={tiles.base} url={tiles.base} attribution={ESRI_ATTRIBUTION} maxNativeZoom={tiles.maxNative} maxZoom={18} />
        {ctxLayer && (
          <ImageOverlay key={`ctx-${ctxLayer.scope}-${ctxLayer.name}`} url={ctxLayer.image_url.replace('image?', 'image?fmt=webp&')} bounds={toBounds(ctxLayer.bounds)} opacity={opacity} zIndex={2} />
        )}
        {showScore && scoreLayer && (
          <ImageOverlay key={`score-${scoreLayer.scope}-${scoreLayer.name}`} url={scoreLayer.image_url} bounds={toBounds(scoreLayer.bounds)} opacity={(ctxLayer ? Math.min(opacity, 0.55) : opacity) * (scope === 'global' && zoom >= 12 ? 0.15 : scope === 'global' && zoom >= 9 ? 0.5 : 1)} zIndex={3} />
        )}
        <Pane name="labels" style={{ zIndex: 450, pointerEvents: 'none' }}>
          <TileLayer key={tiles.labels} url={tiles.labels} pane="labels" maxNativeZoom={tiles.maxNative} maxZoom={18} />
        </Pane>
        {point && <Marker position={[point.lat, point.lon]} icon={markerIcon} />}
        <ClickCatcher onClick={select} onZoom={setZoom} />
      </MapContainer>

      {/* controls */}
      <div className={`pointer-events-none absolute inset-x-0 top-0 z-[1000] flex flex-col gap-3 px-4 pt-24 md:inset-x-auto md:left-5 md:top-0 md:pt-24 ${panelOpen ? "max-md:hidden" : ""}`}>
        <LayerPanel
          catalog={catalog}
          metric={metric} setMetric={setMetric}
          showScore={showScore} setShowScore={setShowScore}
          context={context} setContext={setContext}
          opacity={opacity} setOpacity={setOpacity}
          tn={tn} setTn={setTn}
          baseKind={baseKind} setBaseKind={chooseBase}
          onPlace={onPlace}
        />
      </div>

      {/* zoom */}
      <div className="absolute bottom-8 left-5 z-[1000] hidden flex-col overflow-hidden rounded-2xl bg-[var(--glass)] shadow-[var(--shadow)] ring-1 ring-line backdrop-blur-xl md:flex">
        <button type="button" aria-label="Zoom in" onClick={() => map?.zoomIn()} className="h-10 w-10 text-lg text-ink hover:bg-ink/[0.06]">+</button>
        <span className="h-px bg-line" />
        <button type="button" aria-label="Zoom out" onClick={() => map?.zoomOut()} className="h-10 w-10 text-lg text-ink hover:bg-ink/[0.06]">-</button>
      </div>

      {/* empty-state hint / offline banner */}
      <AnimatePresence>
        {!panelOpen && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 16 }}
            transition={{ duration: 0.5, ease: EASE }}
            className="pointer-events-none absolute inset-x-0 bottom-8 z-[1000] flex flex-col items-center gap-2.5 px-4"
          >
            <FamousParks onPick={onPlace} />
            <div className="glass flex items-center gap-2.5 rounded-full bg-[var(--glass)] px-5 py-3 text-sm shadow-[var(--shadow)] ring-1 ring-line backdrop-blur-xl">
              {offline ? <WarningCircle size={18} weight="light" className="text-warn" /> : <CursorClick size={18} weight="light" className="text-accent" />}
              <span className="text-ink-2">{offline ? 'Analysis server offline. Layers may not load.' : 'Click anywhere on land to analyse it'}</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* result panel: right sheet on desktop, bottom sheet on mobile */}
      <AnimatePresence>
        {panelOpen && (
          <motion.aside
            key="panel"
            initial={{ opacity: 0, x: 40 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 40 }}
            transition={{ duration: 0.55, ease: EASE }}
            className="glass absolute inset-x-2 bottom-2 z-[1001] h-[72dvh] overflow-hidden rounded-[1.75rem] bg-[var(--glass)] shadow-[var(--shadow)] ring-1 ring-line backdrop-blur-2xl md:inset-x-auto md:bottom-5 md:right-5 md:top-24 md:h-auto md:w-[440px]"
            aria-label="Site analysis"
          >
            <ResultPanel
              pred={pred}
              loading={loading}
              error={error}
              point={point}
              inputs={inputs}
              setInputs={setInputs}
              onClose={close}
              onRetry={() => point && analyse(point.lat, point.lon, inputs)}
            />
          </motion.aside>
        )}
      </AnimatePresence>
    </div>
  )
}
