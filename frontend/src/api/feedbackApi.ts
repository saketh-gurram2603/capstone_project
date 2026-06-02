const BASE = '/it-kb'

// ── Types ─────────────────────────────────────────────────────────────────────

export type Sentiment = 'positive' | 'negative'
export type FeedbackStatus = 'PENDING' | 'VERIFIED' | 'DISMISSED'

export interface FeedbackItem {
  feedback_id: string
  session_id: string | null
  query: string
  sentiment: Sentiment
  fix_index: number
  fix_total: number
  resolution_text: string | null
  incident_ids: string[]
  occurrence_count: number
  reason: string | null
  status: FeedbackStatus
  admin_action: string | null
  created_at: string | null
  reviewed_at: string | null
}

export interface FeedbackStats {
  total: number
  negative: number
  positive: number
  pending: number
  verified: number
  dismissed: number
}

export interface FeedbackListResponse {
  total: number
  stats: FeedbackStats
  items: FeedbackItem[]
}

// ── API calls ─────────────────────────────────────────────────────────────────

export async function getFeedback(
  status?: FeedbackStatus,
  sentiment?: Sentiment,
): Promise<FeedbackListResponse> {
  const params = new URLSearchParams()
  if (status) params.set('status', status)
  if (sentiment) params.set('sentiment', sentiment)
  const res = await fetch(`${BASE}/feedback?${params}`)
  if (!res.ok) {
    throw new Error('Failed to fetch feedback')
  }
  return res.json()
}

export async function submitFeedback(
  sessionId: string,
  fixIndex: number,
  sentiment: Sentiment,
  reason?: string,
): Promise<FeedbackItem> {
  const res = await fetch(`${BASE}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      fix_index: fixIndex,
      sentiment,
      reason: reason ?? null,
    }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? 'Failed to submit feedback')
  }
  return res.json()
}

export async function reviewFeedback(
  feedbackId: string,
  status: 'VERIFIED' | 'DISMISSED',
  adminAction?: string,
): Promise<FeedbackItem> {
  const res = await fetch(`${BASE}/feedback/${feedbackId}/review`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status, admin_action: adminAction ?? null }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? 'Failed to update feedback')
  }
  return res.json()
}
