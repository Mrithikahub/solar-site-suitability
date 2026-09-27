import { Warning } from '@phosphor-icons/react'
import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useLocation } from 'react-router-dom'
import { CaseStudy } from '../components/CaseStudy'
import { Bezel, cx, Reveal } from '../components/ui'
import { staticJson } from '../lib/api'

/* eslint-disable @typescript-eslint/no-explicit-any */
type Json = Record<string, any>

const TOC = [
  ['overview', 'Overview'],
  ['data', 'Data sources'],
  ['preprocessing', 'Preprocessing & indices'],
  ['labels', 'Labels & sampling'],
  ['leakage', 'Label leakage'],
  ['ahp', 'AHP baseline'],
  ['ml', 'Machine learning'],
  ['suitability', 'Suitability model'],
  ['regions', 'Performance by region'],
  ['explain', 'Explainability'],
  ['energy', 'Energy estimate'],
  ['case-study', 'High-res case study'],
  ['limits', 'Limitations'],
] as const

const SOURCES = [
  { sensor: 'MODIS Terra', product: 'MOD13A1, MOD09A1, MOD11A2, MCD12Q1, MOD08_M3', bands: 'NDVI, b2 NIR, b6 SWIR, LST_Day, LC_Type1, cloud fraction', res: '500 m - 1°', use: 'Global spectral, thermal, land cover, cloud' },
  { sensor: 'ERA5-Land', product: 'Monthly aggregates, 2005-2024', bands: 'surface_solar_radiation_downwards, temperature_2m', res: '~9 km', use: 'Global irradiance (GHI) and air temperature' },
  { sensor: 'Copernicus DEM', product: 'GLO-30 (TanDEM-X X-band InSAR)', bands: 'DEM', res: '30 m', use: 'Global elevation, slope, aspect (covers > 60° N)' },
  { sensor: 'VIIRS DNB', product: 'Monthly stray-light corrected', bands: 'avg_rad', res: '~460 m', use: 'Night-time lights (development proxy)' },
  { sensor: 'GHSL', product: 'P2023A population', bands: 'population_count', res: '100 m', use: 'Population density (development proxy)' },
  { sensor: 'ESA WorldCover', product: 'v200 (2021)', bands: 'Map', res: '10 m', use: 'Exclusions and display' },
  { sensor: 'Malaria Atlas Project', product: 'Accessibility to cities 2015', bands: 'travel time (min)', res: '~1 km', use: 'Access to demand (development proxy)' },
]

function H2({ id, children }: { id: string; children: React.ReactNode }) {
  return <h2 id={id} className="scroll-mt-28 text-3xl font-semibold tracking-tighter md:text-4xl">{children}</h2>
}

function P({ children }: { children: React.ReactNode }) {
  return <p className="mt-4 max-w-[68ch] leading-relaxed text-ink-2">{children}</p>
}

function Formula({ children, note }: { children: React.ReactNode; note?: string }) {
  return (
    <div className="mt-4 rounded-2xl bg-surface-2 px-5 py-4 ring-1 ring-line">
      <p className="font-mono text-[15px] text-ink">{children}</p>
      {note && <p className="mt-1.5 text-xs text-muted">{note}</p>}
    </div>
  )
}

function Figure({ src, alt, caption, className }: { src: string; alt: string; caption: string; className?: string }) {
  return (
    <figure className={cx('mt-6', className)}>
      <Bezel radius="1.5rem" inner="bg-[#fcfcfb] p-2">
        <img src={src} alt={alt} loading="lazy" className="w-full" />
      </Bezel>
      <figcaption className="mt-2 text-sm text-muted">{caption}</figcaption>
    </figure>
  )
}

function AhpMatrix({ ahp, scope = 'global' }: { ahp: Json; scope?: 'global' | 'tamil_nadu' }) {
  const r = ahp[scope]
  const frac = (v: number) => (v >= 1 ? (Math.abs(v - Math.round(v)) < 1e-6 ? String(Math.round(v)) : v.toFixed(2)) : `1/${Math.round(1 / v)}`)
  return (
    <div className="mt-6">
      <Bezel radius="1.5rem" inner="overflow-x-auto p-2">
        <table className="w-full min-w-[640px] text-sm">
          <thead>
            <tr className="text-xs text-muted">
              <th className="px-3 py-2.5 text-left font-medium">Criterion</th>
              {r.criteria.map((c: string) => <th key={c} className="px-2 py-2.5 text-center font-medium">{c}</th>)}
              <th className="px-3 py-2.5 text-right font-medium">Weight</th>
            </tr>
          </thead>
          <tbody>
            {r.criteria.map((c: string, i: number) => (
              <tr key={c} className={i % 2 === 0 ? 'bg-surface-2/60' : ''}>
                <td className="rounded-l-xl px-3 py-2 text-ink-2">{c}</td>
                {r.matrix[i].map((v: number, j: number) => (
                  <td key={j} className={cx('px-2 py-2 text-center font-mono tabular', i === j ? 'text-muted' : v > 1 ? 'text-accent-ink' : 'text-ink-2')}>{frac(v)}</td>
                ))}
                <td className="rounded-r-xl px-3 py-2 text-right font-mono tabular">{(r.weights[c] * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Bezel>
      <div className="mt-4 grid max-w-xl grid-cols-4 gap-3">
        {[['λmax', r.lambda_max.toFixed(3)], ['CI', r.CI.toFixed(4)], ['RI', String(r.RI)], ['CR', r.CR.toFixed(4)]].map(([k, v]) => (
          <div key={k} className="rounded-xl bg-surface-2 px-3 py-2.5 ring-1 ring-line">
            <p className="text-[11px] text-muted">{k}</p>
            <p className="font-mono tabular">{v}</p>
          </div>
        ))}
      </div>
      <p className="mt-3 text-sm text-ink-2">CR = {r.CR.toFixed(3)} is below 0.10, so the judgements are consistent.</p>
    </div>
  )
}

function MetricsTable({ res, title }: { res: Json; title: string }) {
  const rows: { v: string; m: string; t: Json; cv: number; sp: number; pre?: number }[] = []
  for (const [v, vd] of Object.entries<Json>(res.variants)) {
    for (const [m, r] of Object.entries<Json>(vd.models)) {
      rows.push({ v, m, t: r.test, cv: r.cv_stratified_auc.mean, sp: r.cv_spatial_auc.mean, pre: r.test_on_preconstruction_land?.roc_auc })
    }
  }
  const label: Record<string, string> = { clean: 'Clean', leaky: 'Leaky', as_specified: 'As specified', physical_only: 'Physical only' }
  return (
    <Bezel radius="1.5rem" className="mt-6" inner="overflow-x-auto p-2">
      <table className="w-full min-w-[760px] text-sm">
        <caption className="px-3 pb-1 pt-3 text-left text-sm font-medium">{title}</caption>
        <thead>
          <tr className="text-xs text-muted">
            {['Variant', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'CV AUC', 'Spatial CV', 'AUC pre-constr.'].map((h) => (
              <th key={h} className="px-3 py-2.5 text-left font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={`${r.v}-${r.m}`} className={cx(i % 2 === 0 && 'bg-surface-2/60', r.v === 'clean' && r.m === res.deployed_model && 'font-medium')}>
              <td className="rounded-l-xl px-3 py-2">{label[r.v] ?? r.v}</td>
              <td className="px-3 py-2 text-ink-2">{r.m}{r.v === 'clean' && r.m === res.deployed_model ? ' (deployed)' : ''}</td>
              {[r.t.accuracy, r.t.precision, r.t.recall, r.t.f1, r.t.roc_auc, r.cv, r.sp].map((x, j) => (
                <td key={j} className="px-3 py-2 font-mono tabular">{x.toFixed(3)}</td>
              ))}
              <td className="rounded-r-xl px-3 py-2 font-mono tabular">{r.pre !== undefined ? r.pre.toFixed(3) : 'n/a'}</td>
            </tr>
          ))}
          <tr>
            <td className="rounded-l-xl px-3 py-2">Baseline</td>
            <td className="px-3 py-2 text-ink-2">AHP</td>
            {[res.ahp_test.accuracy, res.ahp_test.precision, res.ahp_test.recall, res.ahp_test.f1, res.ahp_test.roc_auc].map((x: number, j: number) => (
              <td key={j} className="px-3 py-2 font-mono tabular">{x.toFixed(3)}</td>
            ))}
            <td className="px-3 py-2 text-muted">-</td><td className="px-3 py-2 text-muted">-</td><td className="rounded-r-xl px-3 py-2 text-muted">-</td>
          </tr>
        </tbody>
      </table>
    </Bezel>
  )
}

function RegionChart({ rows }: { rows: Json[] }) {
  const data = [...rows].filter((r) => r.auc !== null).sort((a, b) => b.auc - a.auc).map((r) => ({ ...r, label: r.low_sample ? `${r.region} *` : r.region }) as Json)
  return (
    <Bezel radius="1.5rem" className="mt-6" inner="p-6">
      <p className="text-sm font-medium">ROC-AUC by world region</p>
      <p className="text-xs text-muted">Spatially blocked out-of-fold predictions. * fewer than 150 labelled plants: interpret with caution.</p>
      <div className="mt-4" style={{ height: data.length * 34 + 20 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 0, right: 52, left: 0, bottom: 0 }}>
            <CartesianGrid horizontal={false} stroke="var(--line)" />
            <XAxis type="number" domain={[0.5, 1]} tickLine={false} axisLine={false} tick={{ fill: 'var(--muted)', fontSize: 11 }} />
            <YAxis type="category" dataKey="label" width={170} tickLine={false} axisLine={false} tick={{ fill: 'var(--ink-2)', fontSize: 12 }} />
            <Tooltip
              cursor={{ fill: 'var(--surface-2)' }}
              contentStyle={{ background: 'var(--surface)', border: '1px solid var(--line)', borderRadius: 12, fontSize: 12 }}
              formatter={(v, _n, item) => [`${Number(v).toFixed(3)}  (n=${item.payload.n}, plants=${item.payload.n_pos})`, 'AUC']}
            />
            <Bar dataKey="auc" radius={4} barSize={20}>
              {data.map((d) => <Cell key={d.region} fill="#2a78d6" fillOpacity={d.low_sample ? 0.35 : 1} stroke="#2a78d6" strokeDasharray={d.low_sample ? '3 3' : undefined} />)}
              <LabelList dataKey="auc" position="right" formatter={(v: unknown) => Number(v).toFixed(3)} fill="var(--ink-2)" fontSize={11} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Bezel>
  )
}

export default function Methodology() {
  const [m, setM] = useState<Json | null>(null)
  const [ahp, setAhp] = useState<Json | null>(null)
  const [phys, setPhys] = useState<Json | null>(null)
  const [active, setActive] = useState('overview')
  const { hash } = useLocation()

  useEffect(() => {
    if (!hash) return
    const t = window.setTimeout(() => document.getElementById(hash.slice(1))?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 600)
    return () => window.clearTimeout(t)
  }, [hash, m])

  useEffect(() => {
    staticJson<Json>('metrics.json').then(setM).catch(() => setM(null))
    staticJson<Json>('ahp_results.json').then(setAhp).catch(() => setAhp(null))
    staticJson<Json>('physical_model.json').then(setPhys).catch(() => setPhys(null))
  }, [])

  // highlight the TOC entry for the section in view (IntersectionObserver, no scroll listeners)
  useEffect(() => {
    const els = TOC.map(([id]) => document.getElementById(id)).filter(Boolean) as HTMLElement[]
    const io = new IntersectionObserver((entries) => {
      const vis = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)
      if (vis[0]) setActive(vis[0].target.id)
    }, { rootMargin: '-20% 0px -65% 0px' })
    els.forEach((e) => io.observe(e))
    return () => io.disconnect()
  }, [m, ahp, phys])

  return (
    <div className="mx-auto max-w-[1400px] px-4 pb-24 pt-32 md:px-8">
      <Reveal>
        <h1 className="max-w-4xl text-5xl font-semibold leading-[1.02] tracking-tighter md:text-6xl">How the suitability scores are made.</h1>
        <p className="mt-5 max-w-[60ch] text-lg leading-relaxed text-ink-2">Sensors, bands, preprocessing, labels, the AHP baseline, the models and their results, in the order the pipeline runs.</p>
      </Reveal>

      <div className="mt-16 grid gap-12 lg:grid-cols-[220px_1fr]">
        <nav aria-label="Contents" className="hidden lg:block">
          <ul className="sticky top-28 space-y-1 border-l border-line">
            {TOC.map(([id, label]) => (
              <li key={id}>
                <a href={`#${id}`} className={cx('-ml-px block border-l py-1.5 pl-4 text-sm transition-colors duration-300', active === id ? 'border-accent text-ink' : 'border-transparent text-muted hover:text-ink-2')}>{label}</a>
              </li>
            ))}
          </ul>
        </nav>

        <article className="min-w-0 space-y-24">
          <section>
            <H2 id="overview">Overview</H2>
            <P>
              Every location is described by satellite-derived predictors, then scored three ways: an expert-weighted AHP overlay, a supervised
              model of where solar plants have been built (development likelihood), and a physically constrained model of suitability. A
              global model covers all land between 56° S and 72° N. Where 10 m data and local grid maps exist, a high-resolution regional
              model runs as well (see the case study below).
            </P>
            <Figure src="/figures/suitability_map_global.webp" alt="Global physical suitability map" caption="Physical suitability on the 0.5° grid (55,000 land cells), exclusions applied." />
          </section>

          <section>
            <H2 id="data">Data sources</H2>
            <P>All global predictors come from the Google Earth Engine catalog, so one batched request returns every feature for any coordinate.</P>
            <div className="mt-8 grid gap-x-10 gap-y-7 sm:grid-cols-2">
              {SOURCES.map((s) => (
                <div key={s.sensor} className="border-t border-line pt-4">
                  <p className="flex items-baseline justify-between gap-3"><span className="font-medium">{s.sensor}</span><span className="shrink-0 font-mono text-xs text-muted">{s.res}</span></p>
                  <p className="mt-1 text-sm text-ink-2">{s.product}</p>
                  <p className="mt-2 font-mono text-xs text-muted">{s.bands}</p>
                  <p className="mt-2 text-sm text-ink-2">{s.use}</p>
                </div>
              ))}
            </div>
          </section>

          <section>
            <H2 id="preprocessing">Preprocessing & spectral indices</H2>
            <P>
              Sentinel-2 scenes are masked per pixel with Cloud Score+ (cs_cdf at least 0.60) and reduced to a 12-month median, which removes clouds,
              haze and shadows while keeping the typical surface state. Landsat Collection 2 pixels flagged as fill, dilated cloud, cirrus, cloud or
              shadow in QA_PIXEL bits 0-4 are removed. MODIS composites keep good and marginal quality pixels only.
            </P>
            <Formula note="Healthy vegetation reflects strongly in NIR and absorbs red.">NDVI = (NIR - Red) / (NIR + Red) = (B8 - B4) / (B8 + B4)</Formula>
            <Formula note="Built-up and bare surfaces reflect more SWIR than NIR.">NDBI = (SWIR1 - NIR) / (SWIR1 + NIR) = (B11 - B8) / (B11 + B8)</Formula>
            <Formula note="Collection 2 Level 2 scale factors; the USGS single-channel algorithm with ASTER GED emissivity produces ST_B10.">LST [°C] = ST_B10 × 0.00341802 + 149.0 - 273.15</Formula>
            <Formula note="ERA5-Land accumulates downward shortwave radiation per month in J/m².">GHI [kWh/m²/day] = mean monthly SSRD / 3.6×10⁶ / 30.44</Formula>
            <Formula note="Slope-weighted so aspect only matters on sloping ground; flips sign in the southern hemisphere.">Equator-facing index = -cos(aspect) × sign(lat) × min(slope / 10°, 1)</Formula>
            <P>
              Features are sampled at 30 m in batches of 400 points through reduceRegions, with every batch cached on disk. State-wide renders are split
              into Web-Mercator tiles because a single request exceeds Earth Engine's per-request memory.
            </P>
          </section>

          <section>
            <H2 id="labels">Labels & sampling</H2>
            <P>
              Positives come from the Kruitwagen et al. (2021) global inventory of 68,661 PV plants detected in Sentinel-2 and SPOT imagery. One interior
              point is drawn per plant of at least 1 ha, capped at 1,400 plants per world region, giving 8,213 positives. Negatives are half
              area-uniform background land and half hard negatives 5-100 km from a plant, all at least 1 km plus the plant radius from any plant.
            </P>
            <Figure src="/figures/global_training_samples.webp" alt="Map of global training samples" caption="17,213 training points: plants (orange), hard negatives (blue), background negatives (grey)." />
          </section>

          <section>
            <H2 id="leakage">Label leakage</H2>
            <P>
              A satellite looking at an existing plant sees panels, gravel and access roads. ESA WorldCover labels 41% of plant locations as built-up,
              and GHSL even reports people living on some panel fields. Time-varying predictors are therefore taken from before construction
              (2008-10 MODIS globally, 2013-15 Landsat in the high-res case study) and measured now only at prediction time.
            </P>
            <Figure src="/figures/leakage_landcover_then_vs_now.webp" alt="Land cover at plant locations before and after construction" caption="Land cover at plant locations: MODIS 2008 before construction vs ESA WorldCover 2021 after." />
            <Figure src="/figures/leakage_experiment.webp" alt="Leakage experiment" caption="Leaky models win on their own test set but lose on pre-construction land, which is what a new site looks like." />
          </section>

          <section>
            <H2 id="ahp">AHP baseline</H2>
            <P>
              The Analytic Hierarchy Process turns pairwise expert judgements into weights: the principal eigenvector of the comparison matrix. Each
              criterion is rescaled to 0-1 with a linear fuzzy membership function, combined as a weighted sum and multiplied by an exclusion mask for
              water, wetland, built-up land, snow and slopes above 15°.
            </P>
            <Formula note="n = number of criteria; RI is Saaty's random index (1.24 for n = 6, 1.32 for n = 7).">CI = (λmax - n) / (n - 1),   CR = CI / RI,   accept if CR &lt; 0.10</Formula>
            {ahp ? <AhpMatrix ahp={ahp} /> : <div className="skeleton mt-6 h-72" />}
            <Figure src="/figures/ahp_map_global.webp" alt="Global AHP suitability map" caption="AHP suitability on the global grid. It rewards deserts strongly and ignores where development actually happens." />
          </section>

          <section>
            <H2 id="ml">Machine learning</H2>
            <P>
              Random Forest and XGBoost are trained on a stratified 80/20 split, with stratified 5-fold cross-validation on the training data and spatially
              blocked 5-fold cross-validation (2° blocks, or solar sites in the case study) on all data. The spatial score is the honest one: nearby points
              are never split between training and testing.
            </P>
            {m ? (
              <>
                <MetricsTable res={m.global} title="Global model" />
              </>
            ) : <div className="skeleton mt-6 h-96" />}
            <div className="grid gap-5 md:grid-cols-2">
              <Figure src="/figures/roc_global.webp" alt="ROC curves global" caption="ROC curves on the global test set." />
              <Figure src="/figures/confusion_global.webp" alt="Confusion matrix global" caption="Confusion matrix of the deployed development model." />
            </div>
          </section>

          <section>
            <H2 id="suitability">Suitability model</H2>
            <P>
              Plant labels encode economics as well as physics: almost no plants exist in the Sahara because there is no grid or demand nearby. Plain
              supervised learning therefore marks deserts unsuitable. The headline suitability model fixes this in four steps: physical predictors
              only, region-balanced sample weights, positive-unlabelled learning that keeps only reliable negatives (4,610 of 9,000), and monotonic
              constraints for known physical effects. A logit temperature of 2 restores gradation without changing the ranking.
            </P>
            {phys ? (
              <>
                <div className="mt-6 grid gap-3 sm:grid-cols-3">
                  {[
                    ['Before', 'RF, unweighted', phys.before],
                    ['Step 1', 'Region-balanced + monotonic', phys.balanced],
                    ['Final', '+ reliable negatives (PU)', phys.balanced_pu],
                  ].map(([k, d, r]) => (
                    <div key={k as string} className="rounded-2xl bg-surface-2 p-4 ring-1 ring-line">
                      <p className="text-xs text-muted">{k as string}: {d as string}</p>
                      <p className="mt-2 font-mono text-2xl tabular">{(r as Json).test_auc_reliable.toFixed(3)}</p>
                      <p className="text-[11px] text-muted">AUC, plants vs reliable negatives</p>
                      <p className="mt-2 font-mono text-sm text-ink-2 tabular">{(r as Json).test_auc.toFixed(3)}</p>
                      <p className="text-[11px] text-muted">AUC vs all negatives</p>
                    </div>
                  ))}
                </div>
                <p className="mt-3 max-w-[68ch] text-sm text-ink-2">
                  The AUC against all negatives drops by design: open, sunny land without plants now scores as suitable. Sanity check on exact
                  coordinates (one cached Earth Engine sample per site):
                </p>
                <Bezel radius="1.5rem" className="mt-4" inner="overflow-x-auto p-2">
                  <table className="w-full min-w-[560px] text-sm">
                    <thead><tr className="text-xs text-muted">{['Site', 'Expected', 'Before', 'Final', 'Result'].map((h) => <th key={h} className="px-3 py-2.5 text-left font-medium">{h}</th>)}</tr></thead>
                    <tbody>
                      {phys.sanity.map((s: Json, i: number) => (
                        <tr key={s.site} className={i % 2 === 0 ? 'bg-surface-2/60' : ''}>
                          <td className="rounded-l-xl px-3 py-2">{s.site}</td>
                          <td className="px-3 py-2 text-ink-2">{s.expected}</td>
                          <td className="px-3 py-2 font-mono tabular text-ink-2">{Number(s.before).toFixed(1)}</td>
                          <td className="px-3 py-2 font-mono tabular">{Number(s['final + excl.']).toFixed(1)}{s.excluded ? <span className="ml-2 font-sans text-xs text-muted">excluded: {s.excluded}</span> : null}</td>
                          <td className="rounded-r-xl px-3 py-2">{s.verdict === '✓' ? <span className="text-good">Pass</span> : <span className="text-bad">Fail</span>}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Bezel>
              </>
            ) : <div className="skeleton mt-6 h-72" />}
          </section>

          <section>
            <H2 id="regions">Performance by region</H2>
            <P>Development-model accuracy varies with how well each region is represented. Regions with fewer than 150 labelled plants are flagged.</P>
            {m ? <RegionChart rows={m.global.per_region} /> : <div className="skeleton mt-6 h-80" />}
            {m && (
              <div className="mt-4 flex items-start gap-3 rounded-2xl bg-warn/10 px-4 py-3 text-sm ring-1 ring-warn/25">
                <Warning size={18} weight="light" className="mt-0.5 shrink-0 text-warn" />
                <p className="text-ink-2">
                  {m.global.per_region.filter((r: Json) => r.low_sample).map((r: Json) => `${r.region} (${r.n_pos} plants)`).join(' and ')}: too few labels for stable metrics.
                  Predictions there lean on patterns learned elsewhere.
                </p>
              </div>
            )}
          </section>

          <section>
            <H2 id="explain">Explainability</H2>
            <P>
              TreeExplainer computes exact Shapley values, so every score splits into per-feature contributions that sum to the prediction. In the
              map, each site's contributions are rescaled to score points.
            </P>
            <div className="grid gap-5 md:grid-cols-2">
              <Figure src="/figures/shap_summary_global.webp" alt="SHAP summary global development model" caption="Development model: access, lights and population dominate." />
              <Figure src="/figures/feature_importance_global.webp" alt="Feature importance global" caption="Gain-based importance of the development model." />
            </div>
          </section>

          <section>
            <H2 id="energy">Energy estimate</H2>
            <Formula note="A = array area (default 1 acre = 4,047 m²), η = module efficiency (0.20), PR = performance ratio (0.75).">E [kWh/yr] = A × GHI × 365 × η × PR</Formula>
            <P>At 5.4 kWh/m²/day, one acre yields about 1,200 MWh per year. CO₂ savings use the IEA world-average grid intensity of 0.475 kg/kWh.</P>
          </section>

          <section>
            <H2 id="case-study">High-resolution case study (10 m)</H2>
            <div className="mt-6">
              <CaseStudy
                ahpTable={ahp ? <AhpMatrix ahp={ahp} scope="tamil_nadu" /> : undefined}
              />
            </div>
            {m && <MetricsTable res={m.tamil_nadu} title="Regional model (test split by solar site)" />}
          </section>

          <section>
            <H2 id="limits">Limitations</H2>
            <ul className="mt-4 max-w-[68ch] list-disc space-y-2 pl-5 leading-relaxed text-ink-2">
              <li>Grid estimates use the nearest 0.5° cell (up to about 40 km away) when live Earth Engine extraction is unavailable.</li>
              <li>The high-res case-study rasters are decoded from 8-bit renders, so values are quantised to about 1/255 of their display range.</li>
              <li>The inventory ends in 2018 and under-represents Central Asia and Oceania.</li>
              <li>Land ownership, grid capacity, flood risk and protected areas are not modelled.</li>
              <li>The far north-east of Siberia is missing from the global grid because boundary polygons split at the antimeridian.</li>
            </ul>
          </section>
        </article>
      </div>
    </div>
  )
}
