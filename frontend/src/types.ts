export type HardFilterRule = {
  field: string
  operator: string
  value: string | number
  label?: string | null
}

export type BulkModeConfig = {
  enabled: boolean
  llm_top_k: number
  min_pre_score: number
  always_full_below: number
  hybrid_first?: boolean
  phase2_llm_extract?: boolean
}

export type ScoreWeights = {
  ats: number
  embed: number
  llm: number
}

export type Criteria = {
  hiring_prompt: string
  job_description?: string
  shortlist_size: number
  min_final_score: number
  exclude_duplicates_from_shortlist: boolean
  score_weights: ScoreWeights
  hard_filters: HardFilterRule[]
  bulk_mode: BulkModeConfig
}

export type Job = {
  id: number
  name: string
  status: string
  criteria?: Criteria
  total_count?: number
  processed_count?: number
  error_message?: string | null
  created_at?: string
  updated_at?: string
  current_stage?: string | null
  current_filename?: string | null
  control?: string | null
  failed_count?: number
  is_running?: boolean
  shortlist_size?: number
}

export type Progress = {
  job_id: number
  status: string
  processed_count: number
  total_count: number
  percent: number
  is_running: boolean
  error_message?: string | null
  current_stage?: string | null
  current_filename?: string | null
  control?: string
  failed_count?: number
  elapsed_seconds?: number | null
  eta_seconds?: number | null
  phase_done?: number | null
  phase_total?: number | null
  phase_percent?: number | null
}

export type MatchPoints = {
  matched?: string[]
  missing?: string[]
  summary?: string
}

export type StructuredProfile = {
  name?: string | null
  email?: string | null
  phone?: string | null
  location?: string | null
  total_years_experience?: number | null
  notice_period_days?: number | null
  skills?: string[] | string | null
  licenses?: string[] | string | null
  certifications?: string[] | string | null
  education?: Array<Record<string, unknown> | string> | null
  work_history?: Array<Record<string, unknown> | string> | null
  summary?: string | null
  _extraction?: {
    trust_warnings?: string[]
    field_confidence?: Record<string, string>
    overall?: string
  }
  _confidence?: Record<string, string>
  _ranking?: MatchPoints
  [key: string]: unknown
}

export type ResultRow = {
  resume_id: number
  job_id: number
  filename?: string
  source_filename?: string
  candidate_name?: string
  resume_status?: string
  error_message?: string | null
  structured?: StructuredProfile | null
  hr_status?: string
  hr_notes?: string | null
  stage?: string | null
  hard_filter_pass?: number | boolean | null
  hard_filter_results?: unknown
  ats_score?: number | null
  embedding_similarity?: number | null
  llm_score?: number | null
  llm_justification?: string | null
  final_score?: number | null
  rank?: number | null
  shortlist_size?: number
  min_final_score?: number
  shortlisted?: boolean
  is_duplicate?: boolean
  duplicate_of?: number[]
  duplicate_reasons?: string[]
  duplicate_peers?: number[]
  below_min_score?: boolean
  match_points?: MatchPoints
  name_confidence?: string
  name_source?: string
  name_warning?: string
  name_needs_review?: boolean
  seen_elsewhere?: boolean
  seen_in_other_jobs?: Array<{
    job_id: number
    job_name?: string
    resume_id?: number
    candidate_name?: string | null
    final_score?: number | null
    rank?: number | null
    match_reasons?: string[]
  }>
}

export type NameReviewItem = {
  resume_id: number
  rank?: number | null
  candidate_name?: string
  filename?: string
  name_confidence?: string
  name_source?: string
  name_warning?: string
  shortlisted?: boolean
  final_score?: number | null
}

export type Template = {
  id: number
  name: string
  criteria: Criteria
  created_at?: string
}

export type Health = {
  status: string
  ready?: boolean
  ollama?: {
    ok?: boolean
    base?: string
    has_extract_model?: boolean
    has_score_model?: boolean
    has_embed_model?: boolean
    extract_model?: string
    score_model?: string
    embed_model?: string
    error?: string
  }
  config?: {
    api_host?: string
    api_port?: number
    extract_model?: string
    score_model?: string
    embed_model?: string
    default_concurrency?: number
    config_path?: string
  }
  db?: string
}

export type BriefQuality = {
  score: number
  length_ok: boolean
  char_count: number
  checks: Array<{ id: string; label: string; present: boolean }>
  suggestions: string[]
  ready: boolean
}

export type JobError = {
  resume_id: number
  filename?: string
  status?: string
  stage?: string | null
  error_message?: string | null
}
