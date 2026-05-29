const BASE = '/it-kb'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface SearchFilters {
  priority?: string
  impact?: string
  category?: string
}

export interface SearchQuery {
  query: string
  filters?: SearchFilters
  mode?: 'hybrid' | 'semantic'
}

export interface ResolutionOption {
  resolution_text: string
  occurrence_count: number
  source_incident_ids: string[]
}

export interface SearchResultItem {
  incident_id: string
  title: string
  description: string
  resolution_notes: string
  category: string
  priority: string | null
  impact: string | null
  similarity_score: number
  rerank_score: number
}

export interface SearchResponse {
  results: SearchResultItem[]
  resolution_options: ResolutionOption[]
  total_found: number
  adaptive_k_used: number
  retrieval_method: string
  cached: boolean
  latency_ms: number
}

// ── API call ──────────────────────────────────────────────────────────────────

export async function searchIncidents(payload: SearchQuery): Promise<SearchResponse> {
  const res = await fetch(`${BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? 'Search failed')
  }
  return res.json()
}
