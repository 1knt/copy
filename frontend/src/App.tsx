import { lazy, Suspense } from 'react'
import { NavLink, Navigate, Route, Routes, Link } from 'react-router-dom'
import { parseModelInfo } from './api'
import { useResource, useTheme } from './hooks'
import type { Theme } from './hooks'
import { ModelContext } from './model-context'
import { ASSIGNMENT_2_ONLY } from './scope'
import Explore from './pages/Explore'
import Evaluation from './pages/Evaluation'
import About from './pages/About'

const HEALTH_POLL_MS = 15000
const Analyzer = import.meta.env.VITE_SITE_SCOPE === 'a2' ? () => null : lazy(() => import('./pages/Analyzer'))

function ServiceStatus() {
  const live = useResource('health/live', value => value as { status?: string }, { pollMs: HEALTH_POLL_MS })
  const ready = useResource('health/ready', value => value, { pollMs: HEALTH_POLL_MS })

  const online = live.loading && live.data === null && live.error === '' ? null : live.data?.status === 'ok'
  const modelReady = ready.loading && ready.data === null && ready.error === '' ? null : ready.data !== null && ready.error === ''

  return (
    <>
      <div className={online ? 'status-cell push live' : 'status-cell push'}>
        <span className={online ? 'dot on' : 'dot'} aria-hidden="true" />
        <b>{online === null ? 'Checking API' : online ? 'API connected' : 'API unavailable'}</b>
      </div>
      <div className={modelReady === null ? 'status-cell' : modelReady ? 'status-cell live' : 'status-cell warn'}>
        <span className={modelReady ? 'dot on' : 'dot'} aria-hidden="true" />
        <b>{modelReady === null ? 'Checking model' : modelReady ? 'Model connected' : 'Model not connected yet'}</b>
      </div>
    </>
  )
}

const THEMES: { value: Theme; label: string; path: ReturnType<typeof themeIcon> }[] = [
  { value: 'system', label: 'Match system theme', path: themeIcon('system') },
  { value: 'light', label: 'Light theme', path: themeIcon('light') },
  { value: 'dark', label: 'Dark theme', path: themeIcon('dark') },
]

function themeIcon(kind: Theme) {
  if (kind === 'light') {
    return (
      <>
        <circle cx="8" cy="8" r="3.2" />
        <path d="M8 1v1.8M8 13.2V15M1 8h1.8M13.2 8H15M3.1 3.1l1.3 1.3M11.6 11.6l1.3 1.3M12.9 3.1l-1.3 1.3M4.4 11.6l-1.3 1.3" />
      </>
    )
  }
  if (kind === 'dark') return <path d="M13.2 9.6A5.8 5.8 0 0 1 6.4 2.8a5.9 5.9 0 1 0 6.8 6.8Z" />
  return (
    <>
      <rect x="1.6" y="2.6" width="12.8" height="9" rx="1.4" />
      <path d="M5.6 13.4h4.8" />
    </>
  )
}

function ThemeSwitch() {
  const [theme, setTheme] = useTheme()
  return (
    <div className="theme-switch" role="group" aria-label="Colour theme">
      {THEMES.map(option => (
        <button
          key={option.value}
          type="button"
          aria-pressed={theme === option.value}
          aria-label={option.label}
          title={option.label}
          onClick={() => setTheme(option.value)}
        >
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" aria-hidden="true">
            {option.path}
          </svg>
        </button>
      ))}
    </div>
  )
}

function NotFound() {
  return (
    <div className="rail">
      <div className="nameplate">
        <span className="eyebrow micro">ERROR 404</span>
        <h1>This page does not exist.</h1>
        <p className="lede">The address did not match any part of this project.</p>
      </div>
      <div className="datum" />
      <Link className="btn ghost" to="/">{ASSIGNMENT_2_ONLY ? 'Return to the data explorer' : 'Return to the analyzer'}</Link>
    </div>
  )
}

export default function App() {
  const model = useResource('model', parseModelInfo)
  return (
    <ModelContext.Provider value={model}>
    <div className="app-shell">
      <a className="skip-link" href="#main">Skip to content</a>

      <header className="site">
        <div className="rail header-inner">
          <Link to="/" className="brand" aria-label="SMS Spam Screening, home">
            <span className="brand-mark" aria-hidden="true"><i /></span>
            <span>
              <span className="brand-name">SMS Spam Screening</span>
              <span className="brand-sub">COS30049 · HN1_8</span>
            </span>
          </Link>
          <nav className="main" aria-label="Main navigation">
            {ASSIGNMENT_2_ONLY ? (
              <NavLink to="/" end>Explore</NavLink>
            ) : (
              <>
                <NavLink to="/" end>Analyzer</NavLink>
                <NavLink to="/explore">Explore</NavLink>
              </>
            )}
            <NavLink to="/evaluation">Evaluation</NavLink>
            <NavLink to="/about">About</NavLink>
          </nav>
          <ThemeSwitch />
        </div>
      </header>

      <div className="status-rail">
        <div className="rail status-inner">
          <div className="status-cell">{ASSIGNMENT_2_ONLY ? 'ASSIGNMENT 2 · DATA AND MODELS' : 'ASSIGNMENT 3 · FULL-STACK APPLICATION'}</div>
          <ServiceStatus />
        </div>
      </div>

      <main id="main" tabIndex={-1}>
        <Routes>
          {ASSIGNMENT_2_ONLY ? (
            <>
              <Route path="/" element={<Explore />} />
              <Route path="/explore" element={<Navigate to="/" replace />} />
            </>
          ) : (
            <>
              <Route path="/" element={<Suspense fallback={null}><Analyzer /></Suspense>} />
              <Route path="/explore" element={<Explore />} />
            </>
          )}
          <Route path="/evaluation" element={<Evaluation />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>

      <footer className="site">
        <div className="rail footer-inner">
          <span>HN1_8 · Computing Technology Innovation Project</span>
          <span>Educational prototype · English SMS</span>
        </div>
      </footer>
    </div>
    </ModelContext.Provider>
  )
}
