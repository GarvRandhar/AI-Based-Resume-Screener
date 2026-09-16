import { useEffect, useState, type ReactNode } from 'react'
import { API_BASE, getHealth } from '../api'
import type { Health } from '../types'

export function Header() {
  const [health, setHealth] = useState<Health | null | undefined>(undefined)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      const h = await getHealth()
      if (!cancelled) setHealth(h)
    }
    load()
    const id = setInterval(load, 15_000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  const ollama = health?.ollama || {}
  const cfg = health?.config || {}

  let statusBadge: ReactNode
  if (health === undefined) {
    statusBadge = <span className="badge badge-gray">Checking…</span>
  } else if (health === null) {
    statusBadge = (
      <div className="header-actions">
        <span className="badge badge-red">API offline</span>
        <span className="muted" style={{ fontSize: '0.75rem' }}>
          Run <code>python app.py</code>
        </span>
      </div>
    )
  } else if (health.ready) {
    statusBadge = <span className="badge badge-green">System ready</span>
  } else if (ollama.ok) {
    const missing: string[] = []
    if (!ollama.has_extract_model) missing.push(cfg.extract_model || ollama.extract_model || 'LLM')
    if (!ollama.has_embed_model) missing.push(cfg.embed_model || ollama.embed_model || 'embed')
    statusBadge = (
      <div className="header-actions">
        <span className="badge badge-orange">Models missing</span>
        <span className="muted" style={{ fontSize: '0.75rem' }}>
          Pull {missing.join(', ')}
        </span>
      </div>
    )
  } else {
    statusBadge = (
      <div className="header-actions">
        <span className="badge badge-orange">Ollama down</span>
        <span className="muted" style={{ fontSize: '0.75rem' }}>
          {(ollama.error || '').slice(0, 48)}
        </span>
      </div>
    )
  }

  return (
    <>
      <header className="app-header">
        <div className="brand-mark">
          <div className="brand-icon" aria-hidden>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
              <path
                d="M10 4H6a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
              />
              <circle cx="16.5" cy="7.5" r="3.5" stroke="currentColor" strokeWidth="1.6" />
              <path
                d="M19.2 10.2 22 13"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
              />
            </svg>
          </div>
          <div>
            <h1>Resume Screener</h1>
            <p className="subtitle">Local hiring ledger · private on your machine</p>
          </div>
        </div>
        {statusBadge}
      </header>

      {health && !health.ready && (
        <details className="panel" open>
          <summary>System health / models</summary>
          <div className="panel-body health-grid">
            <div>
              API: <code>{API_BASE}</code>
            </div>
            <div>
              Ollama: <code>{ollama.base || '—'}</code>
            </div>
            <div>
              Extract model: <code>{cfg.extract_model || '—'}</code>
            </div>
            <div>
              Score model: <code>{cfg.score_model || ollama.score_model || '—'}</code>
            </div>
            <div>
              Embed model: <code>{cfg.embed_model || '—'}</code>
            </div>
            <div>Has extract: {String(!!ollama.has_extract_model)}</div>
            <div>Has score: {String(ollama.has_score_model ?? '—')}</div>
            <div>Has embed: {String(!!ollama.has_embed_model)}</div>
            {!ollama.has_extract_model && (
              <div style={{ gridColumn: '1 / -1' }}>
                <code>ollama pull {cfg.extract_model || 'llama3.2:3b'}</code>
              </div>
            )}
            {ollama.has_score_model === false && (
              <div style={{ gridColumn: '1 / -1' }}>
                Score model missing (falls back to extract). Optional:{' '}
                <code>ollama pull {cfg.score_model || 'llama3.1:8b'}</code>
              </div>
            )}
            {!ollama.has_embed_model && (
              <div style={{ gridColumn: '1 / -1' }}>
                <code>ollama pull {cfg.embed_model || 'nomic-embed-text'}</code>
              </div>
            )}
          </div>
        </details>
      )}
    </>
  )
}
