import type {
  BriefQuality,
  Criteria,
  Health,
  Job,
  JobError,
  Progress,
  ResultRow,
  StructuredProfile,
  Template,
} from './types'

/** FastAPI base URL — override with VITE_API_BASE if needed. */
export const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined) || 'http://127.0.0.1:8000'

export class ApiError extends Error {
  status: number
  detail: string

  constructor(status: number, detail: string) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

async function parseError(res: Response): Promise<string> {
  try {
    const data = await res.json()
    if (typeof data?.detail === 'string') return data.detail
    if (Array.isArray(data?.detail)) {
      return data.detail.map((d: { msg?: string }) => d.msg || JSON.stringify(d)).join('; ')
    }
    return JSON.stringify(data)
  } catch {
    return res.text().catch(() => res.statusText)
  }
}

export async function api<T = unknown>(
  method: string,
  path: string,
  options?: {
    json?: unknown
    formData?: FormData
    timeoutMs?: number
  },
): Promise<T> {
  const url = `${API_BASE}${path}`
  const controller = new AbortController()
  const timeout = options?.timeoutMs ?? 120_000
  const timer = setTimeout(() => controller.abort(), timeout)

  try {
    const headers: Record<string, string> = {}
    let body: BodyInit | undefined
    if (options?.formData) {
      body = options.formData
    } else if (options?.json !== undefined) {
      headers['Content-Type'] = 'application/json'
      body = JSON.stringify(options.json)
    }

    const res = await fetch(url, {
      method,
      headers,
      body,
      signal: controller.signal,
    })

    if (!res.ok) {
      const detail = await parseError(res)
      if (res.status === 0 || res.type === 'error') {
        throw new ApiError(
          res.status,
          `API offline at ${API_BASE}. In another terminal run: python app.py`,
        )
      }
      throw new ApiError(res.status, `API ${res.status}: ${detail}`)
    }

    if (res.status === 204) return undefined as T
    return (await res.json()) as T
  } catch (err) {
    if (err instanceof ApiError) throw err
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new ApiError(0, `Request timed out after ${timeout / 1000}s`)
    }
    if (err instanceof TypeError) {
      throw new ApiError(
        0,
        `API offline at ${API_BASE}. In another terminal run: python app.py`,
      )
    }
    throw err
  } finally {
    clearTimeout(timer)
  }
}

export async function getHealth(): Promise<Health | null> {
  try {
    return await api<Health>('GET', '/health', { timeoutMs: 4000 })
  } catch {
    return null
  }
}

export async function listJobs(limit = 50): Promise<Job[]> {
  const data = await api<{ jobs: Job[] }>('GET', `/jobs?limit=${limit}`)
  return data.jobs || []
}

export async function createJob(body: {
  name: string
  criteria: Criteria
  concurrency: number
}): Promise<Job> {
  const data = await api<{ job: Job }>('POST', '/jobs', { json: body })
  return data.job
}

export async function uploadResumes(
  jobId: number,
  files: File[],
): Promise<{ uploaded_files: number; resumes_registered: number; total_for_job: number }> {
  const form = new FormData()
  for (const f of files) {
    form.append('files', f, f.name)
  }
  return api('POST', `/jobs/${jobId}/upload`, { formData: form, timeoutMs: 300_000 })
}

export async function startJob(jobId: number, concurrency: number): Promise<void> {
  await api('POST', `/jobs/${jobId}/start`, { json: { concurrency } })
}

export async function getProgress(jobId: number): Promise<Progress> {
  return api('GET', `/jobs/${jobId}/progress`)
}

export async function getResults(jobId: number): Promise<{
  job: Job
  results: ResultRow[]
  shortlist_size: number
  shortlisted_count: number
  name_review_queue?: import('./types').NameReviewItem[]
  name_review_count?: number
}> {
  return api('GET', `/jobs/${jobId}/results`)
}

export async function enrichShortlist(jobId: number): Promise<{
  enriched?: number
  skipped?: number
  failed?: number
  total_shortlist?: number
  attempted?: number
}> {
  return api('POST', `/jobs/${jobId}/enrich-shortlist`, { timeoutMs: 600_000 })
}

export async function pauseJob(jobId: number): Promise<void> {
  await api('POST', `/jobs/${jobId}/pause`)
}

export async function resumeJob(jobId: number): Promise<void> {
  await api('POST', `/jobs/${jobId}/resume`)
}

export async function cancelJob(jobId: number): Promise<void> {
  await api('POST', `/jobs/${jobId}/cancel`)
}

export async function rerunFailed(
  jobId: number,
  concurrency: number,
): Promise<{ count?: number }> {
  return api('POST', `/jobs/${jobId}/rerun-failed`, { json: { concurrency } })
}

export async function getJobErrors(jobId: number): Promise<{ count: number; errors: JobError[] }> {
  return api('GET', `/jobs/${jobId}/errors`)
}

export async function patchShortlistSettings(
  jobId: number,
  body: { shortlist_size?: number; min_final_score?: number },
): Promise<void> {
  await api('PATCH', `/jobs/${jobId}/shortlist-settings`, { json: body })
}

export async function getExportPack(jobId: number): Promise<unknown> {
  return api('GET', `/jobs/${jobId}/export-pack`)
}

export async function getResumeDetail(resumeId: number): Promise<{
  resume: ResultRow & { structured?: StructuredProfile | null }
  score?: ResultRow | null
}> {
  return api('GET', `/resumes/${resumeId}`)
}

export async function patchHrStatus(
  resumeId: number,
  body: { hr_status: string; hr_notes?: string },
): Promise<void> {
  await api('PATCH', `/resumes/${resumeId}/hr`, { json: body })
}

export async function overrideResume(
  resumeId: number,
  body: {
    name?: string | null
    total_years_experience?: number | null
    skills?: string
    rescore?: boolean
    run_llm?: boolean
  },
): Promise<void> {
  await api('PATCH', `/resumes/${resumeId}/override`, {
    json: body,
    timeoutMs: 300_000,
  })
}

export async function listTemplates(): Promise<Template[]> {
  const data = await api<{ templates: Template[] }>('GET', '/templates')
  return data.templates || []
}

export async function getTemplate(id: number): Promise<Template> {
  const data = await api<{ template: Template }>('GET', `/templates/${id}`)
  return data.template
}

export async function saveTemplate(name: string, criteria: Criteria): Promise<void> {
  await api('POST', '/templates', { json: { name, criteria } })
}

export async function analyzeBrief(hiringPrompt: string): Promise<BriefQuality> {
  return api('POST', '/brief/analyze', {
    json: { hiring_prompt: hiringPrompt },
    timeoutMs: 10_000,
  })
}
