const BASE = '/it-kb'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface MetricScore {
  name: string
  score: number
  threshold: number
  passed: boolean
  reason?: string
}

export interface EvalRequest {
  run_llm_judge?: boolean
  run_ir_metrics?: boolean
  dataset_path?: string
}

export interface EvalResult {
  run_id: string
  metrics: MetricScore[]
  overall_passed: boolean
  num_test_cases: number
  latency_ms: number
  timestamp: string
}

export interface LatestMetricsResponse {
  run_id: string | null
  metrics: MetricScore[]
  overall_passed: boolean | null
  timestamp: string | null
  num_test_cases?: number
  message: string
}

// ── API calls ─────────────────────────────────────────────────────────────────

export async function runEvaluation(payload: EvalRequest = {}): Promise<EvalResult> {
  const res = await fetch(`${BASE}/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_ir_metrics: true, run_llm_judge: false, ...payload }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? 'Evaluation failed')
  }
  return res.json()
}

export async function getLatestMetrics(): Promise<LatestMetricsResponse> {
  const res = await fetch(`${BASE}/metrics`)
  if (!res.ok) throw new Error('Failed to fetch metrics')
  return res.json()
}
