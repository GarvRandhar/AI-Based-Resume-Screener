import type { StructuredProfile } from './types'

export function fmtNum(v: unknown, digits = 1): string {
  if (v === null || v === undefined || v === '') return '—'
  const n = Number(v)
  if (Number.isNaN(n)) return String(v)
  return n.toFixed(digits)
}

export function fmtDuration(sec?: number | null): string {
  if (sec == null || !Number.isFinite(sec) || sec < 0) return '—'
  const s = Math.round(sec)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  const r = s % 60
  if (m < 60) return r > 0 ? `${m}m ${r}s` : `${m}m`
  const h = Math.floor(m / 60)
  const rm = m % 60
  return rm > 0 ? `${h}h ${rm}m` : `${h}h`
}

export const STAGE_LABELS: Record<string, string> = {
  starting: 'Starting…',
  embedding_jd: 'Embedding hiring brief',
  parsing: 'Parsing resumes',
  hybrid_extract: 'Extracting candidate details',
  extracting: 'Extracting candidate details',
  filtering: 'Applying filters',
  scoring: 'Scoring candidates',
  llm_scoring: 'LLM scoring top candidates',
  llm_extract: 'Extracting profiles with LLM',
  llm_extract_score: 'LLM extracting + scoring',
  enrich_shortlist: 'Enriching shortlist profiles',
  rerun_failed: 'Re-running failed resumes',
  pause_requested: 'Pausing…',
  resuming: 'Resuming…',
  paused: 'Paused',
  cancel_requested: 'Cancelling…',
  cancelled: 'Cancelled',
  done: 'Done',
  error: 'Error',
}

export function stageLabel(stage?: string | null): string {
  if (!stage) return ''
  return STAGE_LABELS[stage] || stage
}

export function publicProfile(structured: StructuredProfile | null | undefined): Record<string, unknown> {
  if (!structured || typeof structured !== 'object') return {}
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(structured)) {
    if (!k.startsWith('_')) out[k] = v
  }
  return out
}

export function formatEducation(edu: unknown): string {
  if (!edu) return '—'
  if (!Array.isArray(edu)) return String(edu)
  const lines: string[] = []
  for (const e of edu) {
    if (!e || typeof e !== 'object') {
      if (e) lines.push(String(e))
      continue
    }
    const rec = e as Record<string, unknown>
    const degree = String(rec.degree || '').trim()
    const field = String(rec.field || '').trim()
    const inst = String(rec.institution || '').trim()
    const year = rec.year
    if (!degree && !field && !inst) continue
    const parts = [degree, field, inst].filter(Boolean)
    let line = parts.join(' · ')
    if (year != null && year !== '' && year !== 'null' && year !== 'NULL') {
      line = `${line} (${year})`
    }
    if (line && !lines.includes(line)) lines.push(line)
  }
  return lines.length ? lines.join('; ') : '—'
}

export function formatList(items: unknown, limit = 20): string {
  if (!items) return '—'
  if (typeof items === 'string') return items || '—'
  if (Array.isArray(items)) {
    const clean = items.map((x) => String(x).trim()).filter(Boolean)
    if (!clean.length) return '—'
    if (clean.length > limit) {
      return `${clean.slice(0, limit).join(', ')} (+${clean.length - limit} more)`
    }
    return clean.join(', ')
  }
  return String(items)
}

export function downloadBlob(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export function toCsv(rows: Record<string, unknown>[]): string {
  if (!rows.length) return ''
  const keys = Object.keys(rows[0])
  const escape = (v: unknown) => {
    const s = v == null ? '' : String(v)
    if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`
    return s
  }
  const lines = [keys.join(',')]
  for (const row of rows) {
    lines.push(keys.map((k) => escape(row[k])).join(','))
  }
  return lines.join('\n')
}

export function trustWarnings(structured: StructuredProfile | null | undefined): {
  warnings: string[]
  overall?: string
} {
  if (!structured || typeof structured !== 'object') return { warnings: [] }
  const ext = structured._extraction || {}
  const warnings = [...(ext.trust_warnings || [])]
  const conf = ext.field_confidence || structured._confidence || {}
  if (conf.name === 'low' && !warnings.some((w) => w.includes('Name'))) {
    warnings.push('Name is low-confidence — verify identity.')
  }
  if (conf.location === 'low' && !warnings.some((w) => w.includes('Location'))) {
    warnings.push('Location is low-confidence or was cleared.')
  }
  if (!structured.name) {
    warnings.push('No reliable name extracted — check resume file name.')
  }
  return { warnings, overall: ext.overall }
}
