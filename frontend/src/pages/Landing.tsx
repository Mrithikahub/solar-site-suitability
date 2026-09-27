import { ArrowDown, Compass, Database, Plant, Stack, Sun, ThermometerHot, Mountains } from '@phosphor-icons/react'
import { motion, useReducedMotion } from 'motion/react'
import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, LabelList, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { SUIT_STEPS } from '../components/charts'
import { Bezel, EASE, PillButton, Reveal } from '../components/ui'
import { staticJson } from '../lib/api'
import { useTheme } from '../lib/theme'

/* ------------------------------------------------------------------ Hero */
function Hero() {
  const reduce = useReducedMotion()
  const item = (i: number) => ({
    initial: reduce ? false : { opacity: 0, y: 24, filter: 'blur(10px)' },
    animate: { opacity: 1, y: 0, filter: 'blur(0px)' },
    transition: { duration: 1, delay: 0.1 + i * 0.09, ease: EASE },
  })
  return (
    <section className="relative mx-auto grid min-h-[100dvh] max-w-[1400px] items-center gap-12 px-4 pb-16 pt-28 md:grid-cols-[1fr_1.15fr] md:px-8 md:pt-24">
      <div className="max-w-xl">
        <motion.p {...item(0)} className="inline-flex rounded-full bg-accent-soft px-3 py-1 text-[11px] font-medium uppercase tracking-[0.18em] text-accent-ink">
          Remote sensing + machine learning
        </motion.p>
        <motion.h1 {...item(1)} className="mt-6 text-5xl font-semibold leading-[1.02] tracking-tighter md:text-6xl lg:text-[4.2rem]">
          See where solar belongs, anywhere on Earth.
        </motion.h1>
        <motion.p {...item(2)} className="mt-6 max-w-[46ch] text-lg leading-relaxed text-ink-2">
          Satellite imagery and machine learning score land for utility-scale solar, and show the reasons behind every score.
        </motion.p>
        <motion.div {...item(3)} className="mt-9 flex flex-wrap items-center gap-3">
          <PillButton to="/map">Open the map</PillButton>
          <PillButton href="#how" variant="ghost" icon={<ArrowDown size={16} weight="light" />}>How it works</PillButton>
        </motion.div>
      </div>

      <motion.div
        initial={reduce ? false : { opacity: 0, y: 40, scale: 0.97 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 1.2, delay: 0.25, ease: EASE }}
      >
        <Bezel radius="2.25rem" pad="0.5rem" inner="bg-surface-2">
          <div className="relative aspect-[2048/987] w-full">
            <img
              src="/img/layers/physical_global.webp"
              alt="World map of physical solar suitability: deserts, steppe and dry savanna score highest; rainforest, boreal forest and ice score lowest."
              className="absolute inset-0 h-full w-full object-cover"
              fetchPriority="high"
            />
          </div>
          <div className="flex flex-wrap items-center justify-between gap-4 border-t border-line px-5 py-4">
            <div>
              <p className="text-sm font-medium">Physical suitability, 55,000 land cells</p>
              <p className="text-xs text-muted">0.5° grid scored from ERA5-Land, MODIS and Copernicus DEM</p>
            </div>
            <div className="w-44">
              <div className="h-1.5 rounded-full" style={{ background: `linear-gradient(90deg, ${SUIT_STEPS.join(',')})` }} />
              <div className="mt-1 flex justify-between font-mono text-[10px] text-muted"><span>0</span><span>100</span></div>
            </div>
          </div>
        </Bezel>
      </motion.div>
    </section>
  )
}

/* ------------------------------------------------------------------ Stats band */
const STATS = [
  { value: '68,661', label: 'solar plants in the global inventory used for labels' },
  { value: '17,213', label: 'labelled training points across ten world regions' },
  { value: '16', label: 'satellite, climate and map data sources' },
  { value: '0.94', label: 'ROC-AUC of the suitability model, plants vs reliable negatives' },
]

function Stats() {
  return (
    <section className="mx-auto max-w-[1400px] px-4 md:px-8">
      <div className="grid grid-cols-2 gap-y-10 border-y border-line py-12 md:grid-cols-4 md:divide-x md:divide-line">
        {STATS.map((s, i) => (
          <Reveal key={s.value} delay={i * 0.06} className="px-2 md:px-8">
            <p className="font-mono text-4xl font-medium tracking-tight tabular md:text-5xl">{s.value}</p>
            <p className="mt-3 max-w-[26ch] text-sm leading-relaxed text-ink-2">{s.label}</p>
          </Reveal>
        ))}
      </div>
    </section>
  )
}

/* ------------------------------------------------------------------ Problem (editorial split) */
function Problem() {
  return (
    <section className="mx-auto grid max-w-[1400px] items-center gap-12 px-4 py-28 md:grid-cols-12 md:px-8 md:py-36">
      <Reveal className="md:col-span-5">
        <h2 className="text-4xl font-semibold leading-[1.05] tracking-tighter md:text-5xl">Solar needs sun, open land and a grid nearby.</h2>
        <p className="mt-6 max-w-[52ch] leading-relaxed text-ink-2">
          Choosing a site by hand means stacking maps of irradiance, slope, vegetation and land use, then guessing the weights.
          Satellites already measure all of it, everywhere, every few days.
        </p>
        <p className="mt-4 max-w-[52ch] leading-relaxed text-ink-2">
          This project turns those measurements into one score per location and learns the weights from thousands of real solar plants.
        </p>
      </Reveal>
      <Reveal delay={0.1} className="md:col-span-6 md:col-start-7">
        <Bezel radius="2rem" inner="relative aspect-[4/3]">
          <img
            src="/img/layers/truecolor.webp"
            alt="Sentinel-2 true-colour composite of Tamil Nadu, September 2025 to September 2026."
            loading="lazy"
            className="h-full w-full scale-[2.1] object-cover object-[42%_58%]"
          />
        </Bezel>
        <p className="mt-3 text-sm text-muted">Tamil Nadu from Sentinel-2, a cloud-masked median of 2,086 scenes.</p>
      </Reveal>
    </section>
  )
}

/* ------------------------------------------------------------------ How it works (bento of real layers) */
function Tile({ img, alt, title, body, Icon, className, imgClass }: {
  img: string; alt: string; title: string; body: string; Icon: typeof Sun; className?: string; imgClass?: string
}) {
  return (
    <Reveal className={className}>
      <Bezel radius="1.75rem" className="group h-full" inner="flex h-full flex-col">
        <div className="relative min-h-[180px] flex-1 overflow-hidden bg-surface-2">
          <img
            src={img}
            alt={alt}
            loading="lazy"
            className={`absolute inset-0 h-full w-full object-cover transition-transform duration-[1400ms] ease-[cubic-bezier(0.32,0.72,0,1)] group-hover:scale-[1.06] ${imgClass ?? ''}`}
          />
        </div>
        <div className="flex gap-3 p-5">
          <Icon size={22} weight="light" className="mt-0.5 shrink-0 text-accent" />
          <div>
            <h3 className="font-medium">{title}</h3>
            <p className="mt-1 text-sm leading-relaxed text-ink-2">{body}</p>
          </div>
        </div>
      </Bezel>
    </Reveal>
  )
}

function HowItWorks() {
  return (
    <section id="how" className="mx-auto max-w-[1400px] scroll-mt-24 px-4 py-24 md:px-8">
      <Reveal>
        <h2 className="max-w-3xl text-4xl font-semibold leading-[1.05] tracking-tighter md:text-5xl">From orbit to a score in four moves.</h2>
      </Reveal>
      <div className="mt-14 grid auto-rows-[minmax(300px,auto)] grid-cols-1 gap-5 md:grid-cols-6">
        <Tile
          className="md:col-span-4 md:row-span-2"
          img="/img/layers/ndvi.webp"
          imgClass="object-[50%_40%]"
          alt="Sentinel-2 NDVI of Tamil Nadu: forested Western Ghats in dark green, dry plains in tan."
          Icon={Plant}
          title="Observe the surface"
          body="Sentinel-2, Landsat 8/9 and MODIS are cloud-masked and composited into vegetation (NDVI) and built-up (NDBI) indices."
        />
        <Tile
          className="md:col-span-2"
          img="/img/layers/lst.webp"
          alt="Landsat land surface temperature of Tamil Nadu."
          Icon={ThermometerHot}
          title="Measure heat and light"
          body="Thermal bands give land surface temperature. ERA5-Land adds long-term irradiance."
        />
        <Tile
          className="md:col-span-2"
          img="/img/layers/slope.webp"
          alt="Terrain slope of Tamil Nadu from SRTM."
          Icon={Mountains}
          title="Read the terrain"
          body="Slope and aspect from 30 m elevation models decide where panels can sit."
        />
        <Tile
          className="md:col-span-3"
          img="/img/layers/landcover.webp"
          alt="ESA WorldCover land cover of Tamil Nadu."
          Icon={Stack}
          title="Learn from real plants"
          body="XGBoost and Random Forest learn what the land looked like before 8,213 real plants were built."
        />
        <Tile
          className="md:col-span-3"
          img="/img/layers/ml_tn.webp"
          alt="Machine-learning suitability map of Tamil Nadu at 285 m."
          Icon={Compass}
          title="Map every location"
          body="The model scores a 0.5° world grid, and any clicked point live from Earth Engine."
        />
      </div>
    </section>
  )
}

/* ------------------------------------------------------------------ Three scores */
function Scores() {
  return (
    <section className="mx-auto max-w-[1400px] px-4 py-24 md:px-8">
      <Reveal>
        <h2 className="max-w-3xl text-4xl font-semibold leading-[1.05] tracking-tighter md:text-5xl">One place, three honest answers.</h2>
      </Reveal>
      <div className="mt-14 grid gap-5 md:grid-cols-5">
        <Reveal className="md:col-span-3 md:row-span-2">
          <Bezel radius="2rem" className="h-full" inner="relative flex h-full flex-col justify-between overflow-hidden p-8 md:p-10">
            <div
              className="pointer-events-none absolute -right-24 -top-24 h-80 w-80 rounded-full opacity-60 blur-3xl"
              style={{ background: `radial-gradient(circle, ${SUIT_STEPS[3]}55, transparent 70%)` }}
            />
            <div className="relative">
              <Sun size={30} weight="light" className="text-accent" />
              <h3 className="mt-6 text-3xl font-semibold tracking-tight">Suitability</h3>
              <p className="mt-4 max-w-[48ch] leading-relaxed text-ink-2">
                How physically suited the land is to utility-scale PV: sunshine, cloud, slope, vegetation and land cover only.
                This is the headline score on every site.
              </p>
            </div>
            <p className="relative mt-10 max-w-[52ch] text-sm leading-relaxed text-muted">
              Trained with region-balanced weights and positive-unlabelled learning, so sunny land without plants is not marked unsuitable just because nobody has built there yet.
            </p>
          </Bezel>
        </Reveal>
        <Reveal delay={0.08} className="md:col-span-2">
          <Bezel radius="2rem" className="h-full" inner="h-full p-7">
            <h3 className="text-xl font-semibold tracking-tight">Development likelihood</h3>
            <p className="mt-3 text-sm leading-relaxed text-ink-2">
              How closely a site resembles places where plants have historically been built, near roads, grids and demand.
              It describes where developers go, not how good the land is.
            </p>
          </Bezel>
        </Reveal>
        <Reveal delay={0.14} className="md:col-span-2">
          <Bezel radius="2rem" className="h-full" inner="h-full p-7">
            <h3 className="text-xl font-semibold tracking-tight">AHP score</h3>
            <p className="mt-3 text-sm leading-relaxed text-ink-2">
              The classic GIS method: expert-weighted criteria from a pairwise comparison matrix, with water, wetland, urban and steep land excluded.
            </p>
          </Bezel>
        </Reveal>
      </div>
    </section>
  )
}

/* ------------------------------------------------------------------ Leakage insight */
interface LeakRow { name: string; own: number; pre: number }

function Leakage() {
  const [rows, setRows] = useState<{ clean: LeakRow[]; leaky: LeakRow[] } | null>(null)
  useEffect(() => {
    staticJson<Record<string, any>>('metrics.json')
      .then((m) => {
        const get = (scope: 'global' | 'tamil_nadu', v: string) => {
          const r = m[scope].variants[v].models[m[scope].deployed_model]
          return { own: r.test.roc_auc, pre: r.test_on_preconstruction_land.roc_auc }
        }
        const g = { c: get('global', 'clean'), l: get('global', 'leaky') }
        const t = { c: get('tamil_nadu', 'clean'), l: get('tamil_nadu', 'leaky') }
        setRows({
          clean: [{ name: 'Global', ...g.c }, { name: 'Tamil Nadu', ...t.c }],
          leaky: [{ name: 'Global', ...g.l }, { name: 'Tamil Nadu', ...t.l }],
        })
      })
      .catch(() => setRows(null))
  }, [])
  const data = rows
    ? rows.clean.map((c, i) => ({ name: c.name, 'Clean model': +c.pre.toFixed(3), 'Leaky model': +rows.leaky[i].pre.toFixed(3) }))
    : []

  return (
    <section className="border-y border-line bg-surface">
      <div className="mx-auto grid max-w-[1400px] items-center gap-14 px-4 py-28 md:grid-cols-2 md:px-8">
        <Reveal>
          <p className="inline-flex rounded-full bg-accent-soft px-3 py-1 text-[11px] font-medium uppercase tracking-[0.18em] text-accent-ink">Label leakage</p>
          <h2 className="mt-5 text-4xl font-semibold leading-[1.05] tracking-tighter md:text-5xl">We trained on the land before the panels.</h2>
          <p className="mt-6 max-w-[52ch] leading-relaxed text-ink-2">
            Imagery of an existing solar farm shows the panels, and WorldCover maps 41% of plants as built-up.
            A model trained on that learns what a solar farm looks like, not where one should go.
          </p>
          <p className="mt-4 max-w-[52ch] leading-relaxed text-ink-2">
            So every time-varying predictor is taken from before construction. On land that has not been built on yet, the clean model wins.
          </p>
        </Reveal>
        <Reveal delay={0.1}>
          <Bezel radius="2rem" inner="p-6">
            <p className="text-sm font-medium">ROC-AUC on pre-construction land</p>
            <p className="text-xs text-muted">Higher is better. Held-out test sites.</p>
            <div className="mt-4 h-72">
              {rows ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data} margin={{ top: 22, right: 8, left: -12, bottom: 0 }} barGap={4}>
                    <CartesianGrid vertical={false} stroke="var(--line)" />
                    <XAxis dataKey="name" tickLine={false} axisLine={{ stroke: 'var(--line-strong)' }} tick={{ fill: 'var(--ink-2)', fontSize: 12 }} />
                    <YAxis domain={[0.7, 1]} tickLine={false} axisLine={false} tick={{ fill: 'var(--muted)', fontSize: 11 }} />
                    <Tooltip cursor={{ fill: 'var(--surface-2)' }} contentStyle={{ background: 'var(--surface)', border: '1px solid var(--line)', borderRadius: 12, fontSize: 12 }} />
                    <Legend iconType="circle" wrapperStyle={{ fontSize: 12, color: 'var(--ink-2)' }} />
                    <Bar dataKey="Clean model" fill="#2a78d6" radius={[4, 4, 0, 0]}>
                      <LabelList dataKey="Clean model" position="top" fill="var(--ink-2)" fontSize={11} />
                    </Bar>
                    <Bar dataKey="Leaky model" fill="#eb6834" radius={[4, 4, 0, 0]}>
                      <LabelList dataKey="Leaky model" position="top" fill="var(--ink-2)" fontSize={11} />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="skeleton h-full w-full" />
              )}
            </div>
          </Bezel>
        </Reveal>
      </div>
    </section>
  )
}

/* ------------------------------------------------------------------ Stack logo wall */
const LOGOS = [
  ['googleearth', 'Google Earth Engine'], ['nasa', 'NASA POWER'], ['openstreetmap', 'OpenStreetMap'], ['python', 'Python'],
  ['scikitlearn', 'scikit-learn'], ['pandas', 'pandas'], ['numpy', 'NumPy'], ['jupyter', 'Jupyter'], ['fastapi', 'FastAPI'],
  ['react', 'React'], ['typescript', 'TypeScript'], ['vite', 'Vite'], ['tailwindcss', 'Tailwind CSS'], ['leaflet', 'Leaflet'],
] as const

function StackWall() {
  const { theme } = useTheme()
  const color = theme === 'dark' ? 'c3c2b7' : '52514e'
  return (
    <section className="mx-auto max-w-[1400px] px-4 py-24 md:px-8">
      <Reveal>
        <h2 className="text-3xl font-semibold tracking-tighter md:text-4xl">Built on open data and open tools.</h2>
        <p className="mt-3 max-w-[60ch] text-ink-2">Plus XGBoost and SHAP for the models and their explanations, Recharts for charts, and ReportLab for PDF site reports.</p>
      </Reveal>
      <Reveal delay={0.08}>
        <ul className="mt-12 grid grid-cols-4 gap-x-6 gap-y-10 sm:grid-cols-7">
          {LOGOS.map(([slug, name]) => (
            <li key={slug} className="flex items-center justify-center">
              <img
                src={`https://cdn.simpleicons.org/${slug}/${color}`}
                alt={name}
                title={name}
                loading="lazy"
                className="h-8 w-8 opacity-80 transition-opacity duration-300 hover:opacity-100"
              />
            </li>
          ))}
        </ul>
      </Reveal>
    </section>
  )
}

/* ------------------------------------------------------------------ Closing CTA */
function Closing() {
  return (
    <section className="mx-auto max-w-[1400px] px-4 pb-28 md:px-8">
      <Reveal>
        <Bezel radius="2.5rem" pad="0.5rem" inner="relative overflow-hidden px-8 py-16 md:px-16 md:py-24">
          <img src="/img/layers/global_ghi.webp" alt="" aria-hidden className="pointer-events-none absolute inset-0 h-full w-full object-cover opacity-25" />
          <div className="relative max-w-2xl">
            <Database size={28} weight="light" className="text-accent" />
            <h2 className="mt-6 text-4xl font-semibold leading-[1.05] tracking-tighter md:text-5xl">Click anywhere and get a site report.</h2>
            <p className="mt-5 max-w-[50ch] leading-relaxed text-ink-2">
              Score, SHAP explanation, remote-sensing features and an energy estimate, downloadable as a PDF.
            </p>
            <div className="mt-9">
              <PillButton to="/map">Open the map</PillButton>
            </div>
          </div>
        </Bezel>
      </Reveal>
    </section>
  )
}

export default function Landing() {
  return (
    <>
      <Hero />
      <Stats />
      <Problem />
      <HowItWorks />
      <Scores />
      <Leakage />
      <StackWall />
      <Closing />
    </>
  )
}
