import { useCallback, useEffect, useState } from 'react'
import { Header } from './components/Header'
import { ResultsView } from './components/ResultsView'
import { SetupView } from './components/SetupView'

const JOB_KEY = 'screener_active_job_id'
const VIEW_KEY = 'screener_view'

type View = 'setup' | 'results'

function loadView(): View {
  try {
    const v = sessionStorage.getItem(VIEW_KEY)
    if (v === 'results' && sessionStorage.getItem(JOB_KEY)) return 'results'
  } catch {
    /* ignore */
  }
  return 'setup'
}

function loadJobId(): number | null {
  try {
    const raw = sessionStorage.getItem(JOB_KEY)
    if (!raw) return null
    const n = Number(raw)
    return Number.isFinite(n) ? n : null
  } catch {
    return null
  }
}

export default function App() {
  const [view, setView] = useState<View>(loadView)
  const [activeJobId, setActiveJobId] = useState<number | null>(loadJobId)

  useEffect(() => {
    try {
      sessionStorage.setItem(VIEW_KEY, view)
      if (activeJobId != null) sessionStorage.setItem(JOB_KEY, String(activeJobId))
      else sessionStorage.removeItem(JOB_KEY)
    } catch {
      /* ignore */
    }
  }, [view, activeJobId])

  const openJob = useCallback((jobId: number) => {
    setActiveJobId(jobId)
    setView('results')
  }, [])

  const newScreening = useCallback(() => {
    setActiveJobId(null)
    setView('setup')
  }, [])

  return (
    <div className="app-shell">
      <Header />
      <div className="view-enter" key={view === 'results' && activeJobId != null ? `r-${activeJobId}` : 'setup'}>
        {view === 'results' && activeJobId != null ? (
          <ResultsView jobId={activeJobId} onBack={newScreening} />
        ) : (
          <SetupView onStarted={openJob} onOpenJob={openJob} />
        )}
      </div>
      <footer style={{ textAlign: 'center', padding: '1.5rem', fontSize: '0.85rem', color: 'var(--text-secondary, #666)' }}>
        By Garv Randhar
      </footer>
    </div>
  )
}
