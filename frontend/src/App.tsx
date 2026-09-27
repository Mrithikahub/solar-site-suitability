import { lazy, Suspense, useEffect } from 'react'
import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { Footer } from './components/Footer'
import { Nav } from './components/Nav'
import { Skeleton } from './components/ui'

const Landing = lazy(() => import('./pages/Landing'))
const MapPage = lazy(() => import('./pages/MapPage'))
const Compare = lazy(() => import('./pages/Compare'))
const Methodology = lazy(() => import('./pages/Methodology'))
const NotFound = lazy(() => import('./pages/NotFound'))

const TITLES: Record<string, string> = {
  '/': 'Solarsite - Satellite solar site suitability',
  '/map': 'Map - Solarsite',
  '/compare': 'Compare sites - Solarsite',
  '/methodology': 'Methodology - Solarsite',
}

function PageFallback() {
  return (
    <div className="mx-auto max-w-7xl px-4 pt-32 md:px-8">
      <Skeleton className="h-12 w-2/3 max-w-xl" />
      <Skeleton className="mt-4 h-5 w-1/2 max-w-md" />
      <Skeleton className="mt-10 h-[50vh] w-full rounded-[2rem]" />
    </div>
  )
}

export default function App() {
  const { pathname } = useLocation()
  const isMap = pathname === '/map'

  useEffect(() => {
    document.title = TITLES[pathname] ?? 'Solarsite'
    window.scrollTo({ top: 0 })
  }, [pathname])

  return (
    <div className="grain min-h-[100dvh] bg-bg text-ink">
      <Nav />
      <main>
        <Suspense fallback={<PageFallback />}>
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/map" element={<MapPage />} />
            <Route path="/compare" element={<Compare />} />
            {/* the regional case study now lives inside the methodology page */}
            <Route path="/tamil-nadu" element={<Navigate to="/methodology#case-study" replace />} />
            <Route path="/methodology" element={<Methodology />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Suspense>
      </main>
      {!isMap && <Footer />}
    </div>
  )
}
