import { List, Moon, Sun, X } from '@phosphor-icons/react'
import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useState } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { useTheme } from '../lib/theme'
import { cx, EASE, IconButton } from './ui'

const LINKS = [
  { to: '/map', label: 'Map' },
  { to: '/compare', label: 'Compare' },
  { to: '/methodology', label: 'Methodology' },
]

export function Logo({ className }: { className?: string }) {
  return (
    <Link to="/" className={cx('flex items-center gap-2.5 font-semibold tracking-tight', className)} aria-label="Solarsite home">
      <span className="relative flex h-7 w-7 items-center justify-center rounded-[9px] bg-ink">
        <span className="h-3 w-3 rounded-full bg-accent" />
        <span className="absolute inset-[5px] rounded-full ring-1 ring-accent/40" />
      </span>
      <span className="text-[15px]">Solarsite</span>
    </Link>
  )
}

/** Floating "island" navigation pill, detached from the top edge. */
export function Nav() {
  const { theme, toggle } = useTheme()
  const [open, setOpen] = useState(false)
  const loc = useLocation()
  useEffect(() => setOpen(false), [loc.pathname])
  useEffect(() => {
    document.body.style.overflow = open ? 'hidden' : ''
    return () => {
      document.body.style.overflow = ''
    }
  }, [open])

  return (
    <>
      <header className="pointer-events-none fixed inset-x-0 top-0 z-40 flex justify-center px-4 pt-4 md:pt-5">
        <nav
          className="glass pointer-events-auto flex h-14 w-full max-w-[880px] items-center justify-between gap-2 rounded-full bg-[var(--glass)] pl-5 pr-2 shadow-[var(--shadow)] ring-1 ring-line backdrop-blur-xl"
          aria-label="Main"
        >
          <Logo />
          <div className="hidden items-center gap-1 md:flex">
            {LINKS.map((l) => (
              <NavLink
                key={l.to}
                to={l.to}
                className={({ isActive }) =>
                  cx(
                    'rounded-full px-3.5 py-1.5 text-[14px] transition-colors duration-300',
                    isActive ? 'bg-ink/[0.07] text-ink dark:bg-white/[0.08]' : 'text-ink-2 hover:text-ink',
                  )
                }
              >
                {l.label}
              </NavLink>
            ))}
          </div>
          <div className="flex items-center gap-1">
            <IconButton label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'} onClick={toggle}>
              <AnimatePresence mode="wait" initial={false}>
                <motion.span
                  key={theme}
                  initial={{ rotate: -90, opacity: 0, scale: 0.6 }}
                  animate={{ rotate: 0, opacity: 1, scale: 1 }}
                  exit={{ rotate: 90, opacity: 0, scale: 0.6 }}
                  transition={{ duration: 0.35, ease: EASE }}
                  className="flex"
                >
                  {theme === 'dark' ? <Sun size={18} weight="light" /> : <Moon size={18} weight="light" />}
                </motion.span>
              </AnimatePresence>
            </IconButton>
            <Link
              to="/map"
              className="hidden rounded-full bg-ink px-4 py-2 text-[13px] font-medium text-bg transition-transform duration-500 ease-[cubic-bezier(0.32,0.72,0,1)] active:scale-[0.97] md:inline-flex"
            >
              Open the map
            </Link>
            <IconButton label={open ? 'Close menu' : 'Open menu'} className="md:hidden" onClick={() => setOpen((o) => !o)}>
              {open ? <X size={20} weight="light" /> : <List size={20} weight="light" />}
            </IconButton>
          </div>
        </nav>
      </header>

      <AnimatePresence>
        {open && (
          <motion.div
            className="glass fixed inset-0 z-30 flex flex-col justify-center bg-[var(--glass)] px-8 backdrop-blur-3xl md:hidden"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.4, ease: EASE }}
          >
            <ul className="space-y-2">
              {[{ to: '/', label: 'Home' }, ...LINKS].map((l, i) => (
                <li key={l.to} className="overflow-hidden">
                  <motion.div
                    initial={{ y: 48, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ duration: 0.6, delay: 0.08 + i * 0.05, ease: EASE }}
                  >
                    <NavLink to={l.to} className="block py-2 text-4xl font-semibold tracking-tight">
                      {l.label}
                    </NavLink>
                  </motion.div>
                </li>
              ))}
            </ul>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
