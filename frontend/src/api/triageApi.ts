const BASE = '/it-kb'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface TriageRequest {
  description: string
  impact?: 'High' | 'Medium' | 'Low'
  urgency?: 'High' | 'Medium' | 'Low'
}

export interface TriageResult {
  escalation_level: 'L1' | 'L2' | 'L3'
  priority: string | null
  confidence: number
  final_answer: string
  l1_summary: string | null
  l2_synthesis: string | null
  escalation_reason: string | null
  escalation_ticket_id: string | null
  model_used: string
  fallback_used: boolean
  latency_ms: number
}

export interface EscalationTicket {
  ticket_id: string
  description: string
  impact: string | null
  urgency: string | null
  l1_summary: string | null
  l2_analysis: string | null
  escalation_reason: string
  status: 'OPEN' | 'IN_PROGRESS' | 'RESOLVED'
  created_at: string
}

export interface EscalationListResponse {
  total: number
  tickets: EscalationTicket[]
}

// ── API calls ─────────────────────────────────────────────────────────────────

export async function runTriage(payload: TriageRequest): Promise<TriageResult> {
  const res = await fetch(`${BASE}/triage`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? 'Triage failed')
  }
  return res.json()
}

export interface ResolveTicketResponse {
  ticket_id: string
  new_incident_id: string
  status: string
  ingested_to_kb: boolean
  message: string
}

export async function resolveEscalation(
  ticketId: string,
  resolutionSteps: string,
): Promise<ResolveTicketResponse> {
  const res = await fetch(`${BASE}/escalations/${ticketId}/resolve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ resolution_steps: resolutionSteps }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? 'Failed to resolve ticket')
  }
  return res.json()
}

export async function getEscalations(
  status?: string,
  limit = 50,
): Promise<EscalationListResponse> {
  const params = new URLSearchParams({ limit: String(limit) })
  if (status) params.set('status', status)
  const res = await fetch(`${BASE}/escalations?${params}`)
  if (!res.ok) {
    throw new Error('Failed to fetch escalations')
  }
  return res.json()
}
