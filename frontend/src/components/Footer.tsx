import { Link } from 'react-router-dom'
import { Logo } from './Nav'

export function Footer() {
  return (
    <footer className="border-t border-line">
      <div className="mx-auto grid max-w-7xl gap-10 px-4 py-14 md:grid-cols-[1.4fr_1fr_1fr] md:px-8">
        <div className="space-y-4">
          <Logo />
          <p className="max-w-sm text-sm leading-relaxed text-ink-2">
            Solar PV site suitability from satellite remote sensing and machine learning. A Remote Sensing open-elective project.
          </p>
        </div>
        <div className="space-y-3 text-sm">
          <p className="font-medium">Explore</p>
          <ul className="space-y-2 text-ink-2">
            <li><Link className="hover:text-ink" to="/map">Map</Link></li>
            <li><Link className="hover:text-ink" to="/compare">Compare sites</Link></li>
            <li><Link className="hover:text-ink" to="/tamil-nadu">Tamil Nadu case study</Link></li>
            <li><Link className="hover:text-ink" to="/methodology">Methodology</Link></li>
          </ul>
        </div>
        <div className="space-y-3 text-sm">
          <p className="font-medium">Data</p>
          <ul className="space-y-2 text-ink-2">
            <li><a className="hover:text-ink" href="https://developers.google.com/earth-engine/datasets" target="_blank" rel="noreferrer">Google Earth Engine catalog</a></li>
            <li><a className="hover:text-ink" href="https://www.nature.com/articles/s41586-021-03957-7" target="_blank" rel="noreferrer">Kruitwagen et al. 2021, Nature</a></li>
            <li><a className="hover:text-ink" href="https://power.larc.nasa.gov/" target="_blank" rel="noreferrer">NASA POWER</a></li>
            <li><a className="hover:text-ink" href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap contributors</a></li>
          </ul>
        </div>
      </div>
      <div className="mx-auto max-w-7xl px-4 pb-10 text-xs text-muted md:px-8">
        Screening tool only. It does not replace a site survey, a grid-capacity study or a land-title check.
      </div>
    </footer>
  )
}
