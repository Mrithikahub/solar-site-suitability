import { PillButton } from '../components/ui'

export default function NotFound() {
  return (
    <section className="mx-auto flex min-h-[80dvh] max-w-3xl flex-col items-start justify-center px-4 md:px-8">
      <p className="font-mono text-sm text-muted">404</p>
      <h1 className="mt-3 text-4xl font-semibold tracking-tighter md:text-6xl">No imagery for this page.</h1>
      <p className="mt-4 max-w-[52ch] text-ink-2">The address does not match any view in the app. The map is a good place to start.</p>
      <div className="mt-8">
        <PillButton to="/map">Open the map</PillButton>
      </div>
    </section>
  )
}
