import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  cancelJob,
  enrichShortlist,
  getExportPack,
  getJobErrors,
  getProgress,
  getResults,
  getResumeDetail,
  overrideResume,
  patchHrStatus,
  patchShortlistSettings,
  pauseJob,
  resumeJob,
  rerunFailed,
} from '../api'
import type {
  Job,
  JobError,
  NameReviewItem,
  Progress,
  ResultRow,
  StructuredProfile,
} from '../types'
import {
  downloadBlob,
  fmtDuration,
  fmtNum,
  formatEducation,
  formatList,
  publicProfile,
  stageLabel,
  toCsv,
  trustWarnings,
} from '../utils'

const POLL_MS = 1500

type Props = {
  jobId: number
  concurrency?: number
  onBack: () => void
}

function RankMedal({ rank }: { rank: number }) {
  const cls = rank === 1 ? 'rank-1' : rank === 2 ? 'rank-2' : rank === 3 ? 'rank-3' : 'rank-n'
  return <span className={`rank-medal ${cls}`}>{rank}</span>
}

function ProfileFields({ pub }: { pub: Record<string, unknown> }) {
  const rows: [string, string][] = [
    ['Name', String(pub.name || '—')],
    ['Email', String(pub.email || '—')],
    ['Phone', String(pub.phone || '—')],
    ['Location', String(pub.location || '—')],
    [
      'Years of experience',
      pub.total_years_experience != null ? String(pub.total_years_experience) : '—',
    ],
    [
      'Notice period (days)',
      pub.notice_period_days != null ? String(pub.notice_period_days) : '—',
    ],
    ['Skills / strengths', formatList(pub.skills)],
    ['Licenses', formatList(pub.licenses)],
    ['Certifications', formatList(pub.certifications)],
    ['Education', formatEducation(pub.education)],
  ]
  const work = (pub.work_history as unknown[]) || []
  const summary = String(pub.summary || '').trim()

  return (
    <div>
      {rows.map(([label, value]) => (
        <div className="kv" key={label}>
          <div className="k">{label}</div>
          <div className="v">{value}</div>
        </div>
      ))}
      {Array.isArray(work) && work.length > 0 && (
        <div className="kv">
          <div className="k">Work history</div>
          {work.slice(0, 6).map((w, i) => {
            if (w && typeof w === 'object') {
              const rec = w as Record<string, unknown>
              const title = rec.title || ''
              const company = rec.company || ''
              const duration = rec.duration || ''
              return (
                <div className="work-item" key={i}>
                  • {String(title)}
                  {company ? ` @ ${company}` : ''}
                  {duration ? ` (${duration})` : ''}
                </div>
              )
            }
            return (
              <div className="work-item" key={i}>
                • {String(w)}
              </div>
            )
          })}
        </div>
      )}
      {summary && (
        <div className="kv">
          <div className="k">Summary</div>
          <div className="v">{summary}</div>
        </div>
      )}
    </div>
  )
}

export function ResultsView({ jobId, concurrency = 4, onBack }: Props) {
  const [progress, setProgress] = useState<Progress | null>(null)
  const [job, setJob] = useState<Job | null>(null)
  const [results, setResults] = useState<ResultRow[]>([])
  const [shortlistSize, setShortlistSize] = useState(5)
  const [error, setError] = useState<string | null>(null)
  const [msg, setMsg] = useState<string | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [tab, setTab] = useState<'rank' | 'review'>('rank')
  const [showOnlyShortlist, setShowOnlyShortlist] = useState(false)
  const [compareIds, setCompareIds] = useState<number[]>([])
  const [errors, setErrors] = useState<JobError[]>([])
  const [editSlSize, setEditSlSize] = useState(5)
  const [editMinScore, setEditMinScore] = useState(0)
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [detailStructured, setDetailStructured] = useState<StructuredProfile | null>(null)
  const [hrNotes, setHrNotes] = useState('')
  const [ovName, setOvName] = useState('')
  const [ovYears, setOvYears] = useState(0)
  const [ovSkills, setOvSkills] = useState('')
  const [actionBusy, setActionBusy] = useState(false)
  const [nameQueue, setNameQueue] = useState<NameReviewItem[]>([])

  const refresh = useCallback(async () => {
    try {
      const [p, payload, errPayload] = await Promise.all([
        getProgress(jobId),
        getResults(jobId),
        getJobErrors(jobId).catch(() => ({ count: 0, errors: [] as JobError[] })),
      ])
      setProgress(p)
      setJob(payload.job)
      setResults(payload.results || [])
      setNameQueue(payload.name_review_queue || [])
      const sl =
        payload.shortlist_size ||
        payload.job?.shortlist_size ||
        (payload.job?.criteria as { shortlist_size?: number } | undefined)?.shortlist_size ||
        5
      setShortlistSize(sl)
      setEditSlSize(sl)
      const min =
        (payload.results?.[0]?.min_final_score as number | undefined) ??
        (payload.job?.criteria as { min_final_score?: number } | undefined)?.min_final_score ??
        0
      setEditMinScore(min)
      setErrors(errPayload.errors || [])
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [jobId])

  useEffect(() => {
    refresh()
  }, [refresh])

  const status = (progress?.status || job?.status || '—').toLowerCase()
  const total = progress?.total_count || 0
  const done = progress?.processed_count || 0
  const isRunning = Boolean(
    progress?.is_running ||
      ['processing', 'pending'].includes(status),
  )

  let displayPercent = 0
  if (status === 'completed') {
    displayPercent = 100
  } else if (progress?.percent != null) {
    displayPercent = isRunning
      ? Math.min(99, Math.max(0, progress.percent))
      : Math.min(100, Math.max(0, progress.percent))
  } else if (total > 0) {
    if (done < total) {
      displayPercent = Math.min(50, Math.round((done / total) * 50))
    } else if (isRunning) {
      displayPercent = 60
    } else {
      displayPercent = 100
    }
  }

  useEffect(() => {
    if (!autoRefresh || !isRunning) return
    const id = setInterval(refresh, POLL_MS)
    return () => clearInterval(id)
  }, [autoRefresh, isRunning, refresh])

  const shortlisted = useMemo(
    () =>
      results
        .filter((r) => r.shortlisted)
        .sort((a, b) => {
          const ra = a.rank ?? 999
          const rb = b.rank ?? 999
          return ra - rb
        }),
    [results],
  )

  const scored = results.filter(
    (r) => Number(r.hard_filter_pass) === 1 || r.hard_filter_pass === true,
  ).length
  const dupCount = results.filter((r) => r.is_duplicate).length
  const rejected = results.filter((r) => r.resume_status === 'rejected').length
  const failed = results.filter((r) => r.resume_status === 'failed').length
  const topScore = results.reduce((m, r) => {
    const s = r.final_score
    if (s == null) return m
    return Math.max(m, Number(s))
  }, 0)

  const displayRows = showOnlyShortlist ? shortlisted : results
  const minFinal =
    results[0]?.min_final_score ??
    (job?.criteria as { min_final_score?: number } | undefined)?.min_final_score ??
    0

  // Selected candidate for review
  useEffect(() => {
    if (!results.length) {
      setSelectedId(null)
      return
    }
    if (selectedId == null || !results.some((r) => r.resume_id === selectedId)) {
      setSelectedId(results[0].resume_id)
    }
  }, [results, selectedId])

  const detail = results.find((r) => r.resume_id === selectedId) || null

  useEffect(() => {
    if (!detail) {
      setDetailStructured(null)
      setHrNotes('')
      return
    }
    setHrNotes(detail.hr_notes || '')
    const structured = detail.structured
    if (structured) {
      setDetailStructured(structured)
      const pub = publicProfile(structured)
      setOvName(String(pub.name || ''))
      setOvYears(Number(pub.total_years_experience) || 0)
      setOvSkills(formatList(pub.skills) === '—' ? '' : formatList(pub.skills))
    } else {
      getResumeDetail(detail.resume_id)
        .then((full) => {
          const s = full.resume?.structured || null
          setDetailStructured(s)
          const pub = publicProfile(s)
          setOvName(String(pub.name || ''))
          setOvYears(Number(pub.total_years_experience) || 0)
          setOvSkills(formatList(pub.skills) === '—' ? '' : formatList(pub.skills))
        })
        .catch(() => setDetailStructured(null))
    }
  }, [detail?.resume_id]) // eslint-disable-line react-hooks/exhaustive-deps

  const statusMeta = (() => {
    if (status === 'completed')
      return { line: 'Ranking complete', color: 'green' as const }
    if (status === 'failed') return { line: 'Ranking failed', color: 'red' as const }
    if (status === 'paused') return { line: 'Paused', color: 'orange' as const }
    if (isRunning) return { line: 'Ranking in progress…', color: 'blue' as const }
    return { line: status.charAt(0).toUpperCase() + status.slice(1), color: 'gray' as const }
  })()

  const stage = progress?.current_stage || job?.current_stage || ''
  const curFile = progress?.current_filename || job?.current_filename || ''
  const stageTxt = stage ? stageLabel(stage) : ''
  const elapsed = progress?.elapsed_seconds
  const eta = progress?.eta_seconds
  const phaseStep =
    progress?.phase_done != null &&
    progress?.phase_total != null &&
    progress.phase_total > 0 &&
    progress.phase_total < total
      ? ` step ${progress.phase_done}/${progress.phase_total}`
      : ''
  let progExtra = ''
  if (isRunning || status === 'processing' || status === 'paused') {
    if (curFile) progExtra += ` · file: ${curFile}`
  }

  const runAction = async (fn: () => Promise<void>, successMsg?: string) => {
    setActionBusy(true)
    setError(null)
    setMsg(null)
    try {
      await fn()
      if (successMsg) setMsg(successMsg)
      await refresh()
    } catch (e) {
      setError(String(e))
    } finally {
      setActionBusy(false)
    }
  }

  const toggleCompare = (id: number) => {
    setCompareIds((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id)
      if (prev.length >= 3) return prev
      return [...prev, id]
    })
  }

  const exportCsv = (shortlistOnly: boolean) => {
    const src = shortlistOnly ? shortlisted : results
    const rows = src.map((r) => {
      const mp = r.match_points || (r.structured as StructuredProfile | undefined)?._ranking || {}
      return {
        Rank: r.hard_filter_pass ? r.rank ?? '' : '',
        Shortlist: r.shortlisted ? 'Yes' : 'No',
        Candidate: r.candidate_name || r.filename || '',
        'Source file': r.source_filename || r.filename || '',
        'Name confidence': r.name_confidence || '',
        Duplicate: r.is_duplicate ? 'Yes' : 'No',
        Status: r.resume_status || '',
        Final: r.final_score ?? '',
        'ATS %': r.ats_score ?? '',
        Embed: r.embedding_similarity ?? '',
        LLM: r.llm_score ?? '',
        Why: r.llm_justification || '',
        'Matched brief points': (mp.matched || []).join('; '),
        'Missing brief points': (mp.missing || []).join('; '),
      }
    })
    const name = shortlistOnly
      ? `shortlist_job_${jobId}.csv`
      : `ranked_job_${jobId}.csv`
    downloadBlob(name, toCsv(rows), 'text/csv')
  }

  const exportPack = async () => {
    try {
      const pack = (await getExportPack(jobId)) as {
        markdown_summary?: string
        [key: string]: unknown
      }
      downloadBlob(
        `export_pack_job_${jobId}.json`,
        JSON.stringify(pack, null, 2),
        'application/json',
      )
      if (pack.markdown_summary) {
        downloadBlob(
          `export_pack_job_${jobId}.md`,
          pack.markdown_summary,
          'text/markdown',
        )
      }
      setMsg('Exported JSON pack + markdown summary')
    } catch (e) {
      setError(String(e))
    }
  }

  const pub = publicProfile(detailStructured)
  const trust = trustWarnings(detailStructured)

  return (
    <div className="stagger">
      <div className="results-top">
        <div>
          <p className="eyebrow">Job #{jobId}</p>
          <h2 className="section-title" style={{ margin: 0 }}>
            Results
          </h2>
          <p className="section-caption" style={{ marginBottom: 0 }}>
            Ranked only against your hiring brief.
          </p>
        </div>
        <button type="button" className="btn" onClick={onBack}>
          ← New screening
        </button>
      </div>

      <label className="toggle-row">
        <input
          type="checkbox"
          checked={autoRefresh}
          onChange={(e) => setAutoRefresh(e.target.checked)}
        />
        Auto-refresh while running
      </label>

      {error && <div className="alert alert-error">{error}</div>}
      {msg && <div className="alert alert-success">{msg}</div>}

      <div className="card glass-strong">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.75rem', flexWrap: 'wrap' }}>
          <span className={`badge badge-${statusMeta.color}`}>{statusMeta.line}</span>
          {isRunning && eta != null && eta > 0 && (
            <span className="badge badge-blue" title="Estimated time remaining">
              ~{fmtDuration(eta)} left
            </span>
          )}
          {isRunning && (
            <span className="muted" style={{ fontSize: '0.8rem' }}>
              Live updates on
            </span>
          )}
        </div>
        <div className="progress-wrap">
          {stageTxt && (isRunning || status === 'processing' || status === 'paused') && (
            <div className="progress-phase">
              <span className={`phase-dot ${status === 'paused' ? 'paused' : ''}`} />
              {stageTxt}
            </div>
          )}
          <div className="progress-bar">
            <span style={{ width: `${displayPercent}%` }} />
          </div>
          <div className="progress-text">
            {stage === 'enrich_shortlist' ? (
              <>Enriching shortlist profiles · {displayPercent.toFixed(0)}%</>
            ) : status === 'completed' ? (
              <>{total} / {total} processed & screened · 100%</>
            ) : done >= total && isRunning ? (
              <>{done} / {total} resumes parsed · Screening candidates ({displayPercent.toFixed(0)}%)</>
            ) : (
              <>{done} / {total} resumes parsed ({displayPercent.toFixed(0)}%)</>
            )}
            {eta != null && eta > 0 ? ` · ~${fmtDuration(eta)} left` : ''}
            {elapsed != null && elapsed > 0 ? ` · ${fmtDuration(elapsed)} elapsed` : ''}
            {phaseStep}
            {progExtra}
          </div>
        </div>
        <div className="btn-row">
          <button
            type="button"
            className="btn btn-sm"
            disabled={!isRunning || actionBusy}
            onClick={() => runAction(() => pauseJob(jobId))}
          >
            Pause
          </button>
          <button
            type="button"
            className="btn btn-sm"
            disabled={actionBusy}
            onClick={() => runAction(() => resumeJob(jobId))}
          >
            Resume
          </button>
          <button
            type="button"
            className="btn btn-sm btn-danger"
            disabled={actionBusy}
            onClick={() => runAction(() => cancelJob(jobId))}
          >
            Cancel
          </button>
          <button
            type="button"
            className="btn btn-sm"
            disabled={actionBusy || isRunning}
            onClick={() =>
              runAction(async () => {
                const r = await rerunFailed(jobId, concurrency)
                setMsg(`Re-running ${r.count ?? 0} failed resume(s)`)
              })
            }
          >
            Re-run failed only
          </button>
          <button
            type="button"
            className="btn btn-sm btn-primary"
            disabled={actionBusy || isRunning || shortlisted.length === 0}
            title="Fill skills & work history for shortlisted candidates via LLM"
            onClick={() =>
              runAction(async () => {
                const r = await enrichShortlist(jobId)
                setMsg(
                  `Shortlist enrich: ${r.enriched ?? 0} updated · ${r.skipped ?? 0} already rich · ${r.failed ?? 0} failed`,
                )
              }, 'Shortlist profiles enriched')
            }
          >
            Enrich shortlist profiles
          </button>
        </div>
      </div>

      {nameQueue.length > 0 && (
        <details className="panel" open>
          <summary>
            Name verify queue ({nameQueue.length}
            {nameQueue.some((n) => n.shortlisted) ? ' · includes shortlist' : ''})
          </summary>
          <div className="panel-body">
            <p className="muted" style={{ marginTop: 0 }}>
              These names look low-confidence or heuristic-only. Open Review, fix the name, and
              save — especially before outreach.
            </p>
            <div className="table-wrap">
              <table className="data">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Shown name</th>
                    <th>File</th>
                    <th>Source</th>
                    <th>Conf.</th>
                    <th>SL</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {nameQueue.map((n) => (
                    <tr key={n.resume_id}>
                      <td>{n.rank ?? '—'}</td>
                      <td>
                        <strong>{n.candidate_name || '—'}</strong>
                        {n.name_warning && (
                          <div className="muted" style={{ fontSize: '0.78rem' }}>
                            {n.name_warning}
                          </div>
                        )}
                      </td>
                      <td className="muted">{n.filename || '—'}</td>
                      <td>
                        <code>{n.name_source || '—'}</code>
                      </td>
                      <td>
                        <span
                          className={`badge ${
                            n.name_confidence === 'high'
                              ? 'badge-green'
                              : n.name_confidence === 'medium'
                                ? 'badge-orange'
                                : 'badge-red'
                          }`}
                        >
                          {n.name_confidence || 'low'}
                        </span>
                      </td>
                      <td>{n.shortlisted ? 'Yes' : 'No'}</td>
                      <td>
                        <button
                          type="button"
                          className="btn btn-sm"
                          onClick={() => {
                            setSelectedId(n.resume_id)
                            setTab('review')
                          }}
                        >
                          Review
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </details>
      )}

      {errors.length > 0 && (
        <details className="panel">
          <summary>Error report ({errors.length} failed)</summary>
          <div className="panel-body">
            {errors.map((e) => (
              <p className="muted" key={e.resume_id} style={{ margin: '0.25rem 0' }}>
                #{e.resume_id} · {e.filename} · {e.stage || ''} · {e.error_message || ''}
              </p>
            ))}
          </div>
        </details>
      )}

      <details className="panel">
        <summary>Shortlist settings (edit without re-running)</summary>
        <div className="panel-body">
          <div className="form-row cols-3-equal">
            <label className="field">
              Shortlist size
              <input
                type="number"
                min={1}
                max={500}
                value={editSlSize}
                onChange={(e) => setEditSlSize(Number(e.target.value) || 1)}
              />
            </label>
            <label className="field">
              Min final score
              <input
                type="number"
                min={0}
                max={100}
                step={5}
                value={editMinScore}
                onChange={(e) => setEditMinScore(Number(e.target.value) || 0)}
              />
            </label>
            <div style={{ display: 'flex', alignItems: 'flex-end' }}>
              <button
                type="button"
                className="btn btn-block"
                onClick={() =>
                  runAction(
                    () =>
                      patchShortlistSettings(jobId, {
                        shortlist_size: editSlSize,
                        min_final_score: editMinScore,
                      }),
                    'Shortlist updated',
                  )
                }
              >
                Apply shortlist settings
              </button>
            </div>
          </div>
        </div>
      </details>

      <div className="btn-row" style={{ margin: '0.75rem 0' }}>
        <button type="button" className="btn btn-sm" onClick={exportPack}>
          Download export pack (JSON)
        </button>
      </div>

      <div className="metrics">
        <div className="metric">
          <div className="label">Status</div>
          <div className="value" style={{ fontSize: '1.1rem' }}>
            {status.charAt(0).toUpperCase() + status.slice(1)}
          </div>
        </div>
        <div className="metric">
          <div className="label">Processed</div>
          <div className="value">
            {done}/{total}
          </div>
        </div>
        <div className="metric">
          <div className="label">Shortlisted</div>
          <div className="value">
            {shortlisted.length} / {shortlistSize}
          </div>
        </div>
        <div className="metric">
          <div className="label">Top score</div>
          <div className="value">{topScore.toFixed(0)}</div>
        </div>
      </div>

      <p className="muted">
        All {scored} scored candidates are ranked. Shortlist = top <strong>{shortlistSize}</strong>
        {minFinal > 0 ? (
          <>
            {' '}
            with Final ≥ <strong>{minFinal.toFixed(0)}</strong>
          </>
        ) : null}{' '}
        (duplicates excluded).
        {dupCount ? ` · Duplicates: ${dupCount}` : ''}
        {rejected ? ` · Rejected: ${rejected}` : ''}
        {failed ? ` · Failed parse: ${failed}` : ''}
      </p>
      {job?.error_message && <div className="alert alert-error">{job.error_message}</div>}

      <div className="tabs">
        <button
          type="button"
          className={`tab ${tab === 'rank' ? 'active' : ''}`}
          onClick={() => setTab('rank')}
        >
          Ranking & shortlist
        </button>
        <button
          type="button"
          className={`tab ${tab === 'review' ? 'active' : ''}`}
          onClick={() => setTab('review')}
        >
          Candidate review
        </button>
      </div>

      {tab === 'rank' && (
        <div className="tab-panel">
          {shortlisted.length > 0 && (
            <>
              <p className="eyebrow">Top candidates</p>
              <h3
                style={{
                  margin: '0 0 0.85rem',
                  fontFamily: 'var(--font-display)',
                  fontSize: '1.45rem',
                  fontWeight: 600,
                }}
              >
                Shortlist (top {shortlistSize})
              </h3>
              <div className="shortlist-grid">
                {shortlisted.slice(0, 3).map((r, i) => {
                  const rank = r.rank || i + 1
                  const just = (r.llm_justification || '').trim()
                  return (
                    <div
                      className="shortlist-card glass-interactive"
                      key={r.resume_id}
                      role="button"
                      tabIndex={0}
                      onClick={() => {
                        setSelectedId(r.resume_id)
                        setTab('review')
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault()
                          setSelectedId(r.resume_id)
                          setTab('review')
                        }
                      }}
                      style={{ cursor: 'pointer' }}
                    >
                      <div>
                        <RankMedal rank={Number(rank)} />
                        <span
                          className="muted"
                          style={{
                            fontSize: '0.72rem',
                            fontWeight: 600,
                            letterSpacing: '0.08em',
                          }}
                        >
                          SHORTLISTED
                        </span>
                      </div>
                      <div className="name">{r.candidate_name || r.filename || 'Unknown'}</div>
                      {(r.source_filename || r.filename) && (
                        <div className="file">{r.source_filename || r.filename}</div>
                      )}
                      <div className="btn-row" style={{ marginTop: '0.25rem', gap: '0.35rem' }}>
                        {r.name_needs_review && (
                          <span className="badge badge-orange">Verify name</span>
                        )}
                        {r.seen_elsewhere && (
                          <span
                            className="badge badge-blue"
                            title={(r.seen_in_other_jobs || [])
                              .map(
                                (p) =>
                                  `${p.job_name || 'Job'} #${p.job_id}${
                                    p.final_score != null ? ` · score ${p.final_score}` : ''
                                  }`,
                              )
                              .join('\n')}
                          >
                            Seen before
                          </span>
                        )}
                        {(() => {
                          const s = r.structured
                          const skills = s?.skills
                          const skillN = Array.isArray(skills)
                            ? skills.length
                            : typeof skills === 'string' && skills.trim()
                              ? 1
                              : 0
                          const workN = Array.isArray(s?.work_history) ? s!.work_history!.length : 0
                          const enriched = Boolean(
                            (s?._extraction as { shortlist_enriched?: boolean } | undefined)
                              ?.shortlist_enriched,
                          )
                          if (enriched) return <span className="badge badge-green">Enriched</span>
                          if (skillN < 2 && workN === 0)
                            return <span className="badge badge-gray">Thin profile</span>
                          return null
                        })()}
                      </div>
                      <div className="score-row">
                        <span className="score-chip">{fmtNum(r.final_score, 1)}</span>
                        <span className="muted" style={{ fontSize: '0.8rem' }}>
                          final
                        </span>
                      </div>
                      <div className="subscores">
                        <span>ATS {fmtNum(r.ats_score, 0)}</span>
                        <span>Emb {fmtNum(r.embedding_similarity, 2)}</span>
                        <span>LLM {fmtNum(r.llm_score, 1)}</span>
                      </div>
                      {just && (
                        <div className="why">
                          {just.slice(0, 200)}
                          {just.length > 200 ? '…' : ''}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
              {shortlisted.length > 3 && (
                <p className="muted">
                  + {shortlisted.length - 3} more in shortlist (see table below)
                </p>
              )}
            </>
          )}
          {status === 'completed' && scored > 0 && shortlisted.length === 0 && (
            <div className="alert alert-info">
              No one marked shortlisted yet. Check ranks — shortlist size is {shortlistSize}.
            </div>
          )}

          <hr className="divider" />
          <p className="eyebrow">Full board</p>
          <h3 style={{ margin: '0 0 0.5rem', fontFamily: 'var(--font-display)', fontSize: '1.45rem', fontWeight: 600 }}>
            All ranked candidates
          </h3>
          <label className="toggle-row">
            <input
              type="checkbox"
              checked={showOnlyShortlist}
              onChange={(e) => setShowOnlyShortlist(e.target.checked)}
            />
            Show shortlisted only
          </label>

          {!results.length ? (
            <div className="alert alert-info">Waiting for the first resumes to finish…</div>
          ) : showOnlyShortlist && !displayRows.length ? (
            <div className="alert alert-warn">No shortlisted candidates to show.</div>
          ) : (
            <div className="table-wrap">
              <table className="data">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Shortlist</th>
                    <th>Candidate</th>
                    <th>File</th>
                    <th>Dup</th>
                    <th>Prior</th>
                    <th>Status</th>
                    <th>Final</th>
                    <th>ATS %</th>
                    <th>Embed</th>
                    <th>LLM</th>
                    <th>Why</th>
                  </tr>
                </thead>
                <tbody>
                  {displayRows.map((r) => (
                    <tr
                      key={r.resume_id}
                      className={selectedId === r.resume_id ? 'selected' : ''}
                      style={{ cursor: 'pointer' }}
                      onClick={() => {
                        setSelectedId(r.resume_id)
                        setTab('review')
                      }}
                    >
                      <td>{r.hard_filter_pass ? r.rank ?? '—' : '—'}</td>
                      <td>{r.shortlisted ? 'Yes' : 'No'}</td>
                      <td>
                        {r.candidate_name || r.filename}
                        {r.name_needs_review && (
                          <span className="badge badge-orange" style={{ marginLeft: 6 }}>
                            name?
                          </span>
                        )}
                      </td>
                      <td className="muted">{r.source_filename || r.filename || ''}</td>
                      <td>{r.is_duplicate ? 'Yes' : ''}</td>
                      <td
                        title={(r.seen_in_other_jobs || [])
                          .map((p) => `${p.job_name} (#${p.job_id})`)
                          .join(', ')}
                      >
                        {r.seen_elsewhere ? 'Yes' : ''}
                      </td>
                      <td>{r.resume_status}</td>
                      <td>{fmtNum(r.final_score, 1)}</td>
                      <td>{fmtNum(r.ats_score, 1)}</td>
                      <td>{fmtNum(r.embedding_similarity, 3)}</td>
                      <td>{fmtNum(r.llm_score, 1)}</td>
                      <td className="why-cell">
                        {(r.llm_justification || '').slice(0, 120)}
                        {(r.llm_justification || '').length > 120 ? '…' : ''}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <hr className="divider" />
          <p className="eyebrow">Side by side</p>
          <h3 style={{ margin: '0 0 0.35rem', fontFamily: 'var(--font-display)', fontSize: '1.45rem', fontWeight: 600 }}>
            Compare candidates
          </h3>
          <p className="muted">Select up to 3 candidates to compare</p>
          <div className="multi-select">
            {results.map((r) => {
              const active = compareIds.includes(r.resume_id)
              return (
                <button
                  key={r.resume_id}
                  type="button"
                  className={`chip-select ${active ? 'active' : ''}`}
                  disabled={!active && compareIds.length >= 3}
                  onClick={() => toggleCompare(r.resume_id)}
                >
                  {r.candidate_name || r.filename} (#{r.resume_id})
                </button>
              )
            })}
          </div>
          {compareIds.length > 0 && (
            <div className={`compare-grid cols-${compareIds.length}`}>
              {compareIds.map((cid) => {
                const r = results.find((x) => x.resume_id === cid)!
                const hr = r.hr_status || 'none'
                const mp = r.match_points || {}
                return (
                  <div className="card glass-interactive" key={cid} style={{ marginTop: 0 }}>
                    <strong style={{ fontFamily: 'var(--font-display)', fontSize: '1.2rem' }}>
                      {r.candidate_name || r.filename}
                    </strong>
                    <p className="muted" style={{ margin: '0.35rem 0' }}>
                      Rank {r.rank ?? '—'} · Final {fmtNum(r.final_score, 1)}
                    </p>
                    <p className="muted" style={{ margin: '0.25rem 0', fontSize: '0.82rem' }}>
                      ATS {fmtNum(r.ats_score, 1)} · Emb {fmtNum(r.embedding_similarity, 2)} · LLM{' '}
                      {fmtNum(r.llm_score, 1)}
                    </p>
                    <p style={{ fontSize: '0.85rem', margin: '0.35rem 0' }}>
                      {(r.llm_justification || '').slice(0, 180)}
                    </p>
                    {hr === 'starred' && <span className="badge badge-orange">Starred</span>}
                    {hr === 'rejected' && <span className="badge badge-red">Rejected</span>}
                    {mp.matched && mp.matched.length > 0 && (
                      <p className="muted" style={{ fontSize: '0.8rem' }}>
                        Matched: {mp.matched.slice(0, 4).join(', ')}
                      </p>
                    )}
                  </div>
                )
              })}
            </div>
          )}

          <div className="btn-row" style={{ marginTop: '1rem' }}>
            <button type="button" className="btn" onClick={() => exportCsv(false)}>
              Download all ranked (CSV)
            </button>
            <button type="button" className="btn" onClick={() => exportCsv(true)}>
              Download shortlist only (top {shortlistSize})
            </button>
          </div>
        </div>
      )}

      {tab === 'review' && (
        <div className="tab-panel">
          {!results.length ? (
            <div className="alert alert-info">Waiting for the first resumes to finish…</div>
          ) : (
            <>
              <p className="eyebrow">Deep dive</p>
              <h3 style={{ margin: '0 0 0.65rem', fontFamily: 'var(--font-display)', fontSize: '1.45rem', fontWeight: 600 }}>
                Candidate detail
              </h3>
              <label className="field">
                Select candidate
                <select
                  value={selectedId ?? ''}
                  onChange={(e) => setSelectedId(Number(e.target.value))}
                >
                  {results.map((r) => (
                    <option key={r.resume_id} value={r.resume_id}>
                      {r.shortlisted ? '★ ' : ''}
                      {(r.hr_status || '') === 'starred' ? '⭐ ' : ''}
                      {(r.hr_status || '') === 'rejected' ? '⛔ ' : ''}
                      {r.candidate_name || r.filename} (#{r.resume_id}
                      {r.rank ? `, rank ${r.rank}` : ''})
                    </option>
                  ))}
                </select>
              </label>

              {detail && (
                <>
                  <div className="card glass-strong" style={{ marginTop: '0.85rem' }}>
                    <p className="muted" style={{ margin: '0 0 0.5rem', fontWeight: 600, letterSpacing: '0.06em', fontSize: '0.72rem', textTransform: 'uppercase' }}>
                      HR actions
                    </p>
                    <div className="btn-row">
                      <button
                        type="button"
                        className="btn btn-sm"
                        onClick={() =>
                          runAction(() =>
                            patchHrStatus(detail.resume_id, {
                              hr_status: 'starred',
                              hr_notes: hrNotes,
                            }),
                          )
                        }
                      >
                        ★ Star
                      </button>
                      <button
                        type="button"
                        className="btn btn-sm btn-danger"
                        onClick={() =>
                          runAction(() =>
                            patchHrStatus(detail.resume_id, {
                              hr_status: 'rejected',
                              hr_notes: hrNotes,
                            }),
                          )
                        }
                      >
                        Reject
                      </button>
                      <button
                        type="button"
                        className="btn btn-sm"
                        onClick={() =>
                          runAction(() =>
                            patchHrStatus(detail.resume_id, {
                              hr_status: 'none',
                              hr_notes: '',
                            }),
                          )
                        }
                      >
                        Clear HR
                      </button>
                    </div>
                    <label className="field" style={{ marginTop: '0.65rem' }}>
                      HR notes
                      <div className="btn-row">
                        <input
                          type="text"
                          value={hrNotes}
                          onChange={(e) => setHrNotes(e.target.value)}
                          style={{ flex: 1 }}
                        />
                        <button
                          type="button"
                          className="btn btn-sm"
                          onClick={() =>
                            runAction(
                              () =>
                                patchHrStatus(detail.resume_id, {
                                  hr_status: detail.hr_status || 'none',
                                  hr_notes: hrNotes,
                                }),
                              'Notes saved',
                            )
                          }
                        >
                          Save notes
                        </button>
                      </div>
                    </label>
                  </div>

                  {detail.shortlisted ? (
                    <div className="alert alert-success">
                      Shortlisted · Rank #{detail.rank} (drive shortlist size: {shortlistSize})
                    </div>
                  ) : detail.rank ? (
                    <div className="alert alert-info">
                      Not shortlisted · Rank #{detail.rank} (only top {shortlistSize} are
                      shortlisted)
                    </div>
                  ) : null}

                  {(detail.name_warning || detail.name_confidence === 'low') && (
                    <div className="alert alert-warn">
                      {detail.name_warning ||
                        'Name is low-confidence — confirm before outreach.'}
                    </div>
                  )}
                  {detail.name_source && detail.name_confidence !== 'low' && (
                    <p className="muted">
                      Name source: {detail.name_source} · confidence:{' '}
                      {detail.name_confidence || 'n/a'}
                    </p>
                  )}

                  <div className="split-2" style={{ marginTop: '0.75rem' }}>
                    <div className="card" style={{ marginTop: 0 }}>
                      <h4>Scores</h4>
                      <div className="metric glass-strong" style={{ marginBottom: '0.75rem' }}>
                        <div className="label">Final</div>
                        <div className="value">{fmtNum(detail.final_score, 1)}</div>
                      </div>
                      <div className="metrics" style={{ margin: 0, gridTemplateColumns: '1fr 1fr 1fr' }}>
                        <div className="metric">
                          <div className="label">ATS %</div>
                          <div className="value" style={{ fontSize: '1.25rem' }}>
                            {fmtNum(detail.ats_score, 1)}
                          </div>
                        </div>
                        <div className="metric">
                          <div className="label">Embed</div>
                          <div className="value" style={{ fontSize: '1.25rem' }}>
                            {fmtNum(detail.embedding_similarity, 3)}
                          </div>
                        </div>
                        <div className="metric">
                          <div className="label">LLM</div>
                          <div className="value" style={{ fontSize: '1.25rem' }}>
                            {fmtNum(detail.llm_score, 1)}
                          </div>
                        </div>
                      </div>
                      <h4 style={{ marginTop: '0.85rem' }}>Why this score</h4>
                      <p style={{ margin: 0, fontSize: '0.9rem' }}>
                        {detail.llm_justification || '—'}
                      </p>
                      {(() => {
                        const mp =
                          detail.match_points ||
                          detailStructured?._ranking ||
                          {}
                        if (!mp.matched && !mp.missing && !mp.summary) return null
                        return (
                          <div style={{ marginTop: '0.75rem' }}>
                            <h4>Brief match checklist</h4>
                            {mp.summary && <p className="muted">{mp.summary}</p>}
                            {mp.matched && mp.matched.length > 0 && (
                              <div className="alert alert-success">
                                Matched: {mp.matched.slice(0, 8).join('; ')}
                              </div>
                            )}
                            {mp.missing && mp.missing.length > 0 && (
                              <div className="alert alert-warn">
                                Not clearly evidenced: {mp.missing.slice(0, 8).join('; ')}
                              </div>
                            )}
                          </div>
                        )
                      })()}
                      {detail.is_duplicate && (
                        <div className="alert alert-error">
                          Possible duplicate (
                          {(detail.duplicate_reasons || ['same contact']).join(', ')}
                          ). Primary peer ids:{' '}
                          {(detail.duplicate_of || []).slice(0, 5).join(', ')}
                        </div>
                      )}
                      {detail.error_message && (
                        <div className="alert alert-warn">{detail.error_message}</div>
                      )}
                    </div>

                    <div className="card" style={{ marginTop: 0 }}>
                      <h4>Profile</h4>
                      {(detail.source_filename || detail.filename) && (
                        <p className="muted">
                          Source file: <code>{detail.source_filename || detail.filename}</code>
                        </p>
                      )}
                      {trust.warnings.length > 0 ? (
                        <div className="alert alert-warn">
                          <strong>Trust flags — review before deciding</strong>
                          <ul style={{ margin: '0.35rem 0 0', paddingLeft: '1.1rem' }}>
                            {trust.warnings.slice(0, 8).map((w) => (
                              <li key={w}>{w}</li>
                            ))}
                          </ul>
                        </div>
                      ) : trust.overall === 'high' ? (
                        <span className="badge badge-green">Extraction: high confidence</span>
                      ) : trust.overall === 'medium' ? (
                        <span className="badge badge-blue">
                          Extraction: medium — spot-check
                        </span>
                      ) : null}
                      {Object.keys(pub).length > 0 ? (
                        <div style={{ marginTop: '0.65rem' }}>
                          <ProfileFields pub={pub} />
                        </div>
                      ) : (
                        <p className="muted">No profile extracted yet.</p>
                      )}
                      <details className="panel">
                        <summary>Technical JSON (for debugging)</summary>
                        <div className="panel-body">
                          {detailStructured ? (
                            <pre className="json-block">
                              {JSON.stringify(detailStructured, null, 2)}
                            </pre>
                          ) : (
                            <p className="muted">No data</p>
                          )}
                        </div>
                      </details>
                    </div>
                  </div>

                  <details className="panel">
                    <summary>Fix fields & re-score (optional)</summary>
                    <div className="panel-body form-grid">
                      <div className="form-row cols-2">
                        <label className="field">
                          Name
                          <input
                            type="text"
                            value={ovName}
                            onChange={(e) => setOvName(e.target.value)}
                          />
                        </label>
                        <label className="field">
                          Years
                          <input
                            type="number"
                            min={0}
                            max={50}
                            step={0.5}
                            value={ovYears}
                            onChange={(e) => setOvYears(Number(e.target.value) || 0)}
                          />
                        </label>
                      </div>
                      <label className="field">
                        Skills (comma-separated)
                        <input
                          type="text"
                          value={ovSkills}
                          onChange={(e) => setOvSkills(e.target.value)}
                        />
                      </label>
                      <button
                        type="button"
                        className="btn btn-primary"
                        disabled={actionBusy}
                        onClick={() =>
                          runAction(
                            () =>
                              overrideResume(detail.resume_id, {
                                name: ovName.trim() || null,
                                total_years_experience: ovYears,
                                skills: ovSkills === '—' ? '' : ovSkills,
                                rescore: true,
                                run_llm: true,
                              }),
                            'Updated',
                          )
                        }
                      >
                        Save & re-score
                      </button>
                    </div>
                  </details>
                </>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
