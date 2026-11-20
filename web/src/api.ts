// Keep every browser request in this typed client.
export const API_BASE_URL = 'http://127.0.0.1:8000'

export type JobStatus = 'queued' | 'running' | 'done' | 'failed'
export type JobStep =
  | 'Extract'
  | 'Check'
  | 'Classify'
  | 'Debate'
  | 'Policy'
  | 'Chart'
  | 'Awaiting approval'

export interface HealthResponse {
  status: 'ok'
  groq_key_configured: boolean
  docker_available: boolean
}

export interface AnalyzeResponse {
  job_id: string
  case_id: string
}

export interface JobResponse {
  job_id: string
  case_id: string
  status: JobStatus
  current_step: JobStep | null
  error_text: string | null
}

export interface CaseSummary {
  id: string
  status: string
  verdict: string
  confidence: string | null
  date: string
}

export interface LinkFinding {
  url: string
  display_url: string
  risk_score: number | null
  reasons: string[]
}

export interface DebateMessage {
  attempt: number
  round: number
  speaker: string
  message: string
}

export interface EvidenceCard {
  rule_id?: string
  case_id?: string
  rule_text?: string
  summary?: string
  why_matches?: string
  why_similar?: string
  score?: number
}

export interface CaseDetail {
  id: string
  source_type: string
  email_text: string
  current_verdict: string
  case_plan: string[]
  extracted_facts: Record<string, unknown>
  sender_domain_findings: string
  link_findings: LinkFinding[]
  auth_results: string[]
  debate_history: DebateMessage[]
  policy_matches: EvidenceCard[]
  similar_past_cases: EvidenceCard[]
  analyst_notes: string[]
  status: string
  cost: Record<string, unknown>
  chart_url: string | null
  report_url: string | null
  created_at: string
  updated_at: string
}

export interface DecisionResponse {
  case_id: string
  status: string
  current_verdict: string
}

export interface CostResponse {
  total_cost: number
  cost_today: number
}

export interface LogCall {
  agent?: string
  model?: string
  prompt_tokens?: number
  completion_tokens?: number
  total_tokens?: number
  tokens?: number
  cost?: number
  time?: string
  timestamp?: string
}

export interface LogsResponse {
  calls: LogCall[]
}

export interface AnalyzeOptions {
  text?: string
  file?: File
  offline: boolean
  noDocker: boolean
  maskPii: boolean
}

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init)
  } catch {
    throw new Error(
      'Backend not running. Start it with: uvicorn api.main:app --port 8000',
    )
  }
  if (!response.ok) {
    let detail = `Request failed (${response.status}).`
    try {
      const body = (await response.json()) as { detail?: unknown }
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      // Keep the readable status message when the server does not return JSON.
    }
    throw new Error(detail)
  }
  return (await response.json()) as T
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request('/api/health', { signal })
}

export function startAnalysis(options: AnalyzeOptions): Promise<AnalyzeResponse> {
  const body = new FormData()
  if (options.file) body.append('file', options.file)
  else body.append('text', options.text ?? '')
  body.append('offline', String(options.offline))
  body.append('no_docker', String(options.noDocker))
  body.append('mask_pii', String(options.maskPii))
  return request('/api/analyze', { method: 'POST', body })
}

export function getJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<JobResponse> {
  return request(`/api/jobs/${encodeURIComponent(jobId)}`, { signal })
}

export function watchJob(
  jobId: string,
  onUpdate: (job: JobResponse) => void,
  signal: AbortSignal,
): Promise<JobResponse> {
  return new Promise((resolve, reject) => {
    const poll = async (): Promise<void> => {
      if (signal.aborted) {
        reject(new DOMException('Request aborted', 'AbortError'))
        return
      }
      try {
        const job = await getJob(jobId, signal)
        onUpdate(job)
        if (job.status === 'done' || job.status === 'failed') {
          resolve(job)
          return
        }
        window.setTimeout(() => void poll(), 2000)
      } catch (error) {
        reject(error)
      }
    }
    void poll()
  })
}

export function getCase(
  caseId: string,
  signal?: AbortSignal,
): Promise<CaseDetail> {
  return request(`/api/cases/${encodeURIComponent(caseId)}`, { signal })
}

export function getCases(signal?: AbortSignal): Promise<CaseSummary[]> {
  return request('/api/cases', { signal })
}

export function approveCase(caseId: string): Promise<DecisionResponse> {
  return request(`/api/cases/${encodeURIComponent(caseId)}/approve`, {
    method: 'POST',
  })
}

export function rejectCase(
  caseId: string,
  reason: string,
): Promise<AnalyzeResponse> {
  return request(`/api/cases/${encodeURIComponent(caseId)}/reject`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ reason }),
  })
}

export function addCaseNote(
  caseId: string,
  note: string,
): Promise<DecisionResponse> {
  return request(`/api/cases/${encodeURIComponent(caseId)}/note`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ note }),
  })
}

export function saveCase(caseId: string): Promise<DecisionResponse> {
  return request(`/api/cases/${encodeURIComponent(caseId)}/save`, {
    method: 'POST',
  })
}

export function getReport(
  caseId: string,
  signal?: AbortSignal,
): Promise<string> {
  return requestText(`/api/cases/${encodeURIComponent(caseId)}/report`, signal)
}

export function getLogs(
  limit = 20,
  signal?: AbortSignal,
): Promise<LogsResponse> {
  return request(`/api/logs?limit=${limit}`, { signal })
}

export function getCosts(signal?: AbortSignal): Promise<CostResponse> {
  return request('/api/costs', { signal })
}

export function assetUrl(path: string): string {
  return `${API_BASE_URL}${path}`
}

async function requestText(
  path: string,
  signal?: AbortSignal,
): Promise<string> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { signal })
  } catch {
    throw new Error(
      'Backend not running. Start it with: uvicorn api.main:app --port 8000',
    )
  }
  if (!response.ok) {
    let detail = `Request failed (${response.status}).`
    try {
      const body = (await response.json()) as { detail?: unknown }
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      // Keep the readable status message when the server does not return JSON.
    }
    throw new Error(detail)
  }
  return response.text()
}
