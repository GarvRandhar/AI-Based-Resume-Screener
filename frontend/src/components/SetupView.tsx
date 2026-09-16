import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  analyzeBrief,
  createJob,
  getTemplate,
  listJobs,
  listTemplates,
  saveTemplate,
  startJob,
  uploadResumes,
} from '../api'
import type { BriefQuality, HardFilterRule, Job, Template } from '../types'

const HARD_FIELDS = [
  { value: 'years_experience', label: 'Years of experience' },
  { value: 'license', label: 'License / registration' },
  { value: 'certification', label: 'Certification' },
  { value: 'degree', label: 'Degree' },
  { value: 'skill', label: 'Skill / capability' },
  { value: 'location', label: 'Location' },
  { value: 'notice_period_days', label: 'Notice period (days)' },
  { value: 'education', label: 'Education (text)' },
]

const HARD_OPS = [
  { value: 'gte', label: '≥ minimum' },
  { value: 'lte', label: '≤ maximum' },
  { value: 'contains', label: 'contains (fuzzy)' },
  { value: 'eq', label: 'equals (fuzzy)' },
  { value: 'in', label: 'one of (comma list)' },
]

const STORAGE_KEY = 'screener_setup_v1'

export type SetupState = {
  driveName: string
  hiringPrompt: string
  shortlistSize: number
  minFinalScore: number
  concurrency: number
  excludeDuplicates: boolean
  bulkEnabled: boolean
  bulkLlmTopK: number
  bulkMinPreScore: number
  bulkAlwaysFullBelow: number
  wAts: number
  wEmbed: number
  wLlm: number
  hardFilters: HardFilterRule[]
}

const DEFAULTS: SetupState = {
  driveName: 'Hiring Drive',
  hiringPrompt: '',
  shortlistSize: 5,
  minFinalScore: 0,
  concurrency: 4,
  excludeDuplicates: true,
  bulkEnabled: true,
  bulkLlmTopK: 20,
  bulkMinPreScore: 30,
  bulkAlwaysFullBelow: 0,
  wAts: 20,
  wEmbed: 30,
  wLlm: 50,
  hardFilters: [],
}

function loadSetup(): SetupState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return { ...DEFAULTS }
    return { ...DEFAULTS, ...JSON.parse(raw) }
  } catch {
    return { ...DEFAULTS }
  }
}

type Props = {
  onStarted: (jobId: number) => void
  onOpenJob: (jobId: number) => void
}

export function SetupView({ onStarted, onOpenJob }: Props) {
  const [form, setForm] = useState<SetupState>(loadSetup)
  const [files, setFiles] = useState<File[]>([])
  const [dragOver, setDragOver] = useState(false)
  const [jobs, setJobs] = useState<Job[]>([])
  const [templates, setTemplates] = useState<Template[]>([])
  const [selectedJob, setSelectedJob] = useState<number | ''>('')
  const [selectedTemplate, setSelectedTemplate] = useState<number | ''>('')
  const [templateName, setTemplateName] = useState('')
  const [briefQ, setBriefQ] = useState<BriefQuality | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)

  // Hard rule draft
  const [hardField, setHardField] = useState('years_experience')
  const [hardOp, setHardOp] = useState('gte')
  const [hardVal, setHardVal] = useState('')

  const set = useCallback(<K extends keyof SetupState>(key: K, value: SetupState[K]) => {
    setForm((prev) => {
      const next = { ...prev, [key]: value }
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
      } catch {
        /* ignore */
      }
      return next
    })
  }, [])

  useEffect(() => {
    listJobs(20)
      .then((j) => {
        setJobs(j)
        if (j.length && selectedJob === '') setSelectedJob(j[0].id)
      })
      .catch(() => setJobs([]))
    listTemplates()
      .then(setTemplates)
      .catch(() => setTemplates([]))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Debounced brief quality
  useEffect(() => {
    const prompt = form.hiringPrompt
    if (!prompt || prompt.trim().length < 10) {
      setBriefQ(null)
      return
    }
    const t = setTimeout(() => {
      analyzeBrief(prompt)
        .then(setBriefQ)
        .catch(() => setBriefQ(null))
    }, 400)
    return () => clearTimeout(t)
  }, [form.hiringPrompt])

  const onFiles = (list: FileList | File[] | null) => {
    if (!list) return
    const arr = Array.from(list)
    setFiles((prev) => {
      const names = new Set(prev.map((f) => f.name + f.size))
      const merged = [...prev]
      for (const f of arr) {
        const key = f.name + f.size
        if (!names.has(key)) {
          merged.push(f)
          names.add(key)
        }
      }
      return merged
    })
  }

  const criteriaPayload = useMemo(() => {
    const wt = Math.max(1, form.wAts + form.wEmbed + form.wLlm)
    return {
      hiring_prompt: form.hiringPrompt.trim(),
      job_description: form.hiringPrompt.trim(),
      shortlist_size: form.shortlistSize,
      min_final_score: form.minFinalScore,
      exclude_duplicates_from_shortlist: form.excludeDuplicates,
      score_weights: {
        ats: form.wAts / wt,
        embed: form.wEmbed / wt,
        llm: form.wLlm / wt,
      },
      hard_filters: form.hardFilters,
      bulk_mode: {
        enabled: form.bulkEnabled,
        llm_top_k: form.bulkLlmTopK,
        min_pre_score: form.bulkMinPreScore,
        always_full_below: form.bulkAlwaysFullBelow,
        hybrid_first: true,
        phase2_llm_extract: false,
      },
    }
  }, [form])

  const addHardRule = () => {
    if (!hardVal.trim()) {
      setError('Enter a value for the hard rule.')
      return
    }
    let val: string | number = hardVal.trim()
    if (
      (hardField === 'years_experience' || hardField === 'notice_period_days') &&
      ['gte', 'lte', 'eq'].includes(hardOp)
    ) {
      const n = hardVal.includes('.') ? parseFloat(hardVal) : parseInt(hardVal, 10)
      if (!Number.isNaN(n)) val = n
    }
    const rule: HardFilterRule = {
      field: hardField,
      operator: hardOp,
      value: val,
      label: `${hardField} ${hardOp} ${val}`,
    }
    set('hardFilters', [...form.hardFilters, rule])
    setHardVal('')
    setError(null)
  }

  const loadTemplateIntoForm = async () => {
    if (selectedTemplate === '') return
    try {
      const t = await getTemplate(selectedTemplate)
      const crit = t.criteria || ({} as Template['criteria'])
      setForm((prev) => {
        const sw = crit.score_weights || {}
        const next: SetupState = {
          ...prev,
          hiringPrompt: crit.hiring_prompt || crit.job_description || '',
          shortlistSize: Number(crit.shortlist_size) || 5,
          minFinalScore: Number(crit.min_final_score) || 0,
          hardFilters: listHard(crit.hard_filters),
          excludeDuplicates: crit.exclude_duplicates_from_shortlist !== false,
          wAts: Math.round(Number(sw.ats ?? 0.2) * 100),
          wEmbed: Math.round(Number(sw.embed ?? 0.3) * 100),
          wLlm: Math.round(Number(sw.llm ?? 0.5) * 100),
          bulkEnabled: crit.bulk_mode?.enabled !== false,
          bulkLlmTopK: crit.bulk_mode?.llm_top_k ?? 20,
          bulkMinPreScore: crit.bulk_mode?.min_pre_score ?? 30,
          bulkAlwaysFullBelow: crit.bulk_mode?.always_full_below ?? 0,
        }
        localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
        return next
      })
      setSuccess(`Loaded “${t.name}”`)
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }

  const handleSaveTemplate = async () => {
    if (!templateName.trim()) {
      setError('Template name required')
      return
    }
    try {
      await saveTemplate(templateName.trim(), criteriaPayload)
      setTemplates(await listTemplates())
      setSuccess('Template saved')
      setTemplateName('')
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }

  const handleRank = async () => {
    setError(null)
    setSuccess(null)
    const prompt = form.hiringPrompt.trim()
    if (prompt.length < 20) {
      setError('Please paste a fuller hiring brief (role, skills, must-haves).')
      return
    }
    if (!files.length) {
      setError('Please upload at least one resume or ZIP.')
      return
    }

    try {
      setBusy('Creating job…')
      const job = await createJob({
        name: form.driveName || 'Hiring Drive',
        criteria: criteriaPayload,
        concurrency: form.concurrency,
      })

      setBusy(`Uploading ${files.length} file(s)…`)
      const up = await uploadResumes(job.id, files)

      setBusy('Starting ranking…')
      await startJob(job.id, form.concurrency)

      setSuccess(
        `Started · ${up.resumes_registered ?? 0} resumes · job #${job.id}`,
      )
      onStarted(job.id)
    } catch (e) {
      setError(String(e))
      setBusy(null)
    }
  }

  return (
    <div className="stagger">
      <p className="eyebrow">New drive</p>
      <h2 className="section-title">Start a screening</h2>
      <p className="section-caption">
        Describe who you want to hire, upload resumes, and get a ranked shortlist — privately,
        on this machine.
      </p>
      <div className="step-pills">
        <span className="step-pill">1 · Brief</span>
        <span className="step-pill">2 · Upload</span>
        <span className="step-pill">3 · Rank</span>
      </div>

      {error && <div className="alert alert-error">{error}</div>}
      {success && <div className="alert alert-success">{success}</div>}
      {busy && (
        <div className="loading-overlay">
          <span className="spinner" /> {busy}
        </div>
      )}

      {jobs.length > 0 && (
        <details className="panel">
          <summary>Open a previous run</summary>
          <div className="panel-body">
            <label className="field">
              Previous job
              <select
                value={selectedJob}
                onChange={(e) =>
                  setSelectedJob(e.target.value ? Number(e.target.value) : '')
                }
              >
                {jobs.map((j) => (
                  <option key={j.id} value={j.id}>
                    #{j.id} · {j.name} · {j.status} · {j.processed_count ?? 0}/
                    {j.total_count ?? 0}
                  </option>
                ))}
              </select>
            </label>
            <div style={{ marginTop: '0.65rem' }}>
              <button
                type="button"
                className="btn btn-block"
                disabled={selectedJob === ''}
                onClick={() => selectedJob !== '' && onOpenJob(selectedJob)}
              >
                Open results
              </button>
            </div>
          </div>
        </details>
      )}

      <details className="panel">
        <summary>Drive templates (save / load brief + settings)</summary>
        <div className="panel-body form-grid">
          {templates.length > 0 && (
            <>
              <label className="field">
                Load template
                <select
                  value={selectedTemplate}
                  onChange={(e) =>
                    setSelectedTemplate(e.target.value ? Number(e.target.value) : '')
                  }
                >
                  <option value="">— none —</option>
                  {templates.map((t) => (
                    <option key={t.id} value={t.id}>
                      {t.name}
                    </option>
                  ))}
                </select>
              </label>
              <button
                type="button"
                className="btn"
                disabled={selectedTemplate === ''}
                onClick={loadTemplateIntoForm}
              >
                Load template into form
              </button>
            </>
          )}
          <label className="field">
            Save current form as template
            <input
              type="text"
              value={templateName}
              onChange={(e) => setTemplateName(e.target.value)}
              placeholder="Template name"
            />
          </label>
          <button type="button" className="btn" onClick={handleSaveTemplate}>
            Save template
          </button>
        </div>
      </details>

      <div className="card glass-strong">
        <h3>Job settings</h3>
        <div className="form-grid">
          <div className="form-row cols-3 job-settings-row">
            <label className="field">
              <span className="field-label">Job name</span>
              <input
                type="text"
                value={form.driveName}
                onChange={(e) => set('driveName', e.target.value)}
                placeholder="e.g. Backend hire — March"
              />
            </label>
            <label
              className="field"
              title="Only the top N ranked candidates are marked Shortlisted"
            >
              <span className="field-label">Shortlist size (top N)</span>
              <input
                type="number"
                min={1}
                max={200}
                value={form.shortlistSize}
                onChange={(e) => set('shortlistSize', Number(e.target.value) || 1)}
              />
            </label>
            <label className="field" title="Minimum final score to enter shortlist (0 = off)">
              <span className="field-label">Min score to shortlist</span>
              <input
                type="number"
                min={0}
                max={100}
                step={5}
                value={form.minFinalScore}
                onChange={(e) => set('minFinalScore', Number(e.target.value) || 0)}
              />
            </label>
          </div>

          <label className="field">
            Hiring brief (required)
            <span className="hint">Primary ranking signal — domain-agnostic</span>
            <textarea
              value={form.hiringPrompt}
              onChange={(e) => set('hiringPrompt', e.target.value)}
              rows={10}
              placeholder={
                'Works for any drive — tech, sales, clinical, operations, campus…\n\n' +
                'Example (sales):\n' +
                'B2B Account Executive, Mumbai, 3+ years enterprise sales.\n' +
                'Must: quota attainment, CRM (Salesforce/HubSpot), strong communication.\n' +
                'Prefer MBA or business degree. Notice under 60 days.'
              }
            />
          </label>

          {briefQ && (
            <details className="panel" open={!briefQ.ready}>
              <summary>
                Hiring brief checklist · quality {briefQ.score}/100
                {briefQ.ready ? ' · ready' : ' · add more detail'}
              </summary>
              <div className="panel-body">
                <ul className="check-list">
                  {briefQ.checks.map((c) => (
                    <li key={c.id} className={c.present ? 'check-ok' : 'check-miss'}>
                      {c.present ? '✓' : '○'} {c.label}
                    </li>
                  ))}
                </ul>
                {briefQ.suggestions.length > 0 && (
                  <>
                    <p className="muted" style={{ fontWeight: 600 }}>
                      Suggestions to improve ranking
                    </p>
                    <ul className="muted" style={{ margin: 0, paddingLeft: '1.1rem' }}>
                      {briefQ.suggestions.map((s) => (
                        <li key={s}>{s}</li>
                      ))}
                    </ul>
                  </>
                )}
              </div>
            </details>
          )}

          <div>
            <div
              className={`file-drop ${dragOver ? 'dragover' : ''}`}
              onDragOver={(e) => {
                e.preventDefault()
                setDragOver(true)
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault()
                setDragOver(false)
                onFiles(e.dataTransfer.files)
              }}
              onClick={() => document.getElementById('resume-input')?.click()}
            >
              <div className="title">Resumes (PDF, DOCX, TXT, or ZIP)</div>
              <div className="hint">Click or drag files here · individual or ZIP</div>
              <input
                id="resume-input"
                type="file"
                multiple
                accept=".pdf,.docx,.doc,.txt,.zip"
                onChange={(e) => onFiles(e.target.files)}
              />
            </div>
            {files.length > 0 && (
              <div className="file-list">
                <span className="badge badge-blue" style={{ marginBottom: '0.4rem' }}>
                  {files.length} file(s) selected
                </span>
                <ul>
                  {files.map((f) => (
                    <li key={f.name + f.size}>
                      {f.name}{' '}
                      <button
                        type="button"
                        className="btn btn-sm btn-ghost"
                        onClick={(e) => {
                          e.stopPropagation()
                          setFiles((prev) => prev.filter((x) => x !== f))
                        }}
                      >
                        Remove
                      </button>
                    </li>
                  ))}
                </ul>
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => setFiles([])}
                  style={{ marginTop: '0.35rem' }}
                >
                  Clear all
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      <details className="panel">
        <summary>Optional hard rules (only if HR needs a hard cut)</summary>
        <div className="panel-body">
          <p className="muted">
            Leave empty for pure prompt ranking. Use for must-pass gates like min years or a
            required license (fuzzy match).
          </p>
          <div className="form-row cols-3-equal">
            <label className="field">
              Field
              <select value={hardField} onChange={(e) => setHardField(e.target.value)}>
                {HARD_FIELDS.map((f) => (
                  <option key={f.value} value={f.value}>
                    {f.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              Operator
              <select value={hardOp} onChange={(e) => setHardOp(e.target.value)}>
                {HARD_OPS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              Value
              <input
                type="text"
                value={hardVal}
                onChange={(e) => setHardVal(e.target.value)}
                placeholder="e.g. 3 or RN"
              />
            </label>
          </div>
          <button type="button" className="btn" style={{ marginTop: '0.65rem' }} onClick={addHardRule}>
            Add hard rule
          </button>
          {form.hardFilters.length === 0 ? (
            <p className="muted">No hard rules — recommended for most drives.</p>
          ) : (
            <ul className="rule-list">
              {form.hardFilters.map((rule, i) => (
                <li key={`${rule.label}-${i}`}>
                  <span>
                    {rule.label}{' '}
                    <code>
                      {rule.field} {rule.operator} {String(rule.value)}
                    </code>
                  </span>
                  <button
                    type="button"
                    className="btn btn-sm"
                    onClick={() =>
                      set(
                        'hardFilters',
                        form.hardFilters.filter((_, idx) => idx !== i),
                      )
                    }
                  >
                    Remove
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </details>

      <details className="panel">
        <summary>Advanced engine options</summary>
        <div className="panel-body form-grid">
          <label className="field">
            LLM concurrency: {form.concurrency}
            <input
              type="range"
              min={1}
              max={8}
              value={form.concurrency}
              onChange={(e) => set('concurrency', Number(e.target.value))}
            />
            <span className="muted" style={{ fontSize: '0.85rem' }}>
              Parallel Ollama chat calls (extract + fit). Embeds use a separate pool.
            </span>
          </label>
          <p className="muted" style={{ margin: 0, fontWeight: 600 }}>
            Score weights (relative; normalized automatically)
          </p>
          <div className="form-row cols-3-equal">
            <label className="field">
              ATS keywords %: {form.wAts}
              <input
                type="range"
                min={0}
                max={100}
                value={form.wAts}
                onChange={(e) => set('wAts', Number(e.target.value))}
              />
            </label>
            <label className="field">
              Embedding similarity: {form.wEmbed}
              <input
                type="range"
                min={0}
                max={100}
                value={form.wEmbed}
                onChange={(e) => set('wEmbed', Number(e.target.value))}
              />
            </label>
            <label className="field">
              LLM fit score: {form.wLlm}
              <input
                type="range"
                min={0}
                max={100}
                value={form.wLlm}
                onChange={(e) => set('wLlm', Number(e.target.value))}
              />
            </label>
          </div>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={form.excludeDuplicates}
              onChange={(e) => set('excludeDuplicates', e.target.checked)}
            />
            Exclude duplicate email/phone from shortlist (keep highest score)
          </label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={form.bulkEnabled}
              onChange={(e) => set('bulkEnabled', e.target.checked)}
            />
            Fast bulk mode (LLM only top candidates)
          </label>
          {form.bulkEnabled && (
            <div className="form-row cols-2">
              <label className="field">
                LLM top-K
                <input
                  type="number"
                  min={1}
                  max={500}
                  value={form.bulkLlmTopK}
                  onChange={(e) => set('bulkLlmTopK', Number(e.target.value) || 1)}
                />
              </label>
              <label className="field">
                Min pre-score
                <input
                  type="number"
                  min={0}
                  max={100}
                  value={form.bulkMinPreScore}
                  onChange={(e) => set('bulkMinPreScore', Number(e.target.value) || 0)}
                />
              </label>
            </div>
          )}
        </div>
      </details>

      <div style={{ marginTop: '1.35rem' }}>
        <button
          type="button"
          className="btn btn-primary btn-block btn-lg"
          onClick={handleRank}
          disabled={!!busy}
        >
          {busy ? (
            <>
              <span className="spinner" /> Ranking…
            </>
          ) : (
            <>Rank resumes</>
          )}
        </button>
      </div>
    </div>
  )
}

function listHard(v: unknown): HardFilterRule[] {
  if (!Array.isArray(v)) return []
  return v as HardFilterRule[]
}
