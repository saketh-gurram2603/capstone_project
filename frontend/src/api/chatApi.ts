const BASE = '/it-kb'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ChatRequest {
  session_id: string | null
  message: string
}

export interface OptionProgress {
  current: number
  total: number
}

export interface ChatResponse {
  session_id: string
  message: string
  role: 'assistant'
  option_progress: OptionProgress | null
  is_escalated: boolean
  escalation_ticket_id: string | null
  all_options_exhausted: boolean
  suggested_actions: string[]
}

// ── API call ──────────────────────────────────────────────────────────────────

export async function sendChatMessage(payload: ChatRequest): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? 'Chat request failed')
  }
  return res.json()
}
