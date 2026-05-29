const BASE = '/it-kb'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface IngestResponse {
  status: string
  message: string
  ingested: number
  skipped: number
  duration_ms: number
  pii_masked_total?: number
}

export interface IngestStatusResponse {
  status: string          // idle | running | completed | failed
  total: number
  ingested: number
  skipped: number
  started_at: string | null
  completed_at: string | null
  duration_ms: number | null
  error: string | null
}

// ── API calls ─────────────────────────────────────────────────────────────────

export async function uploadFile(file: File): Promise<IngestResponse> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/ingest`, { method: 'POST', body: form })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? 'Ingestion failed')
  }
  return res.json()
}

export async function getIngestStatus(): Promise<IngestStatusResponse> {
  const res = await fetch(`${BASE}/ingest/status`)
  if (!res.ok) throw new Error('Failed to fetch ingestion status')
  return res.json()
}
