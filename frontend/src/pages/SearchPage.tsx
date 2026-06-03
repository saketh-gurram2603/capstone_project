import { useState, useRef } from 'react'
import { useMutation } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Search, SlidersHorizontal, Zap, Database, Clock, X,
  ChevronRight, Sparkles, ArrowRight, AlertTriangle,
  CheckCircle, Loader,
} from 'lucide-react'
import { searchIncidents, type SearchFilters, type SearchResponse } from '../api/searchApi'
import { runTriage, type TriageResult } from '../api/triageApi'
import IncidentCard from '../components/IncidentCard'
import ResolutionPanel from '../components/ResolutionPanel'

// If the top search result scores below this, show the low-confidence banner
const LOW_CONFIDENCE_THRESHOLD = 0.50

// ── Filter config ─────────────────────────────────────────
const PRIORITY_CHIPS = [
  { value: 'P1', label: 'P1 Critical', cls: 'badge-critical' },
  { value: 'P2', label: 'P2 High',     cls: 'badge-high'     },
  { value: 'P3', label: 'P3 Medium',   cls: 'badge-medium'   },
  { value: 'P4', label: 'P4 Low',      cls: 'badge-low'      },
]
const CATEGORY_CHIPS = ['Database','Network','Application','Storage','Security','Hardware','Performance']
const IMPACT_CHIPS   = ['High','Medium','Low']
const EXAMPLES = [
  'Database connection pool exhausted under load',
  'VPN users cannot authenticate remotely',
  'Memory usage spikes causing application restart',
  'Disk write latency increased by 10x overnight',
]

function Chip({
  label, active, onClick, cls = '',
}: { label: string; active: boolean; onClick: () => void; cls?: string }) {
  return (
    <button
      onClick={onClick}
      className={`text-[11px] font-semibold px-2.5 py-1 rounded-lg border transition-all duration-120 cursor-pointer
        ${active
          ? `${cls || 'badge-blue'} scale-[1.02]`
          : 'border-[rgba(255,255,255,0.07)] text-[#9CA7B3] hover:border-[rgba(255,255,255,0.14)] hover:text-[#F5F7FA]'
        }`}
      style={{ background: active ? undefined : 'transparent' }}
    >
      {label}
    </button>
  )
}

function SkeletonCard() {
  return (
    <div className="card-sm shimmer" style={{ height: 90, borderRadius: 14 }} />
  )
}

// ── Right panel — AI Insights ─────────────────────────────
function InsightsPanel({ data, onTriage }: {
  data: SearchResponse
  onTriage: (q: string) => void
}) {
  const topScore = data.results[0]?.similarity_score ?? 0
  const categories = data.results.reduce<Record<string, number>>((acc, r) => {
    const cat = r.category ?? 'Unknown'
    acc[cat] = (acc[cat] ?? 0) + 1
    return acc
  }, {})
  const catEntries = Object.entries(categories).sort((a, b) => b[1] - a[1]).slice(0, 5)
  const resolution = data.resolution_options[0]

  return (
    <motion.div
      initial={{ opacity: 0, x: 16 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.25 }}
      className="space-y-3 sticky top-0"
      style={{ minWidth: 240, maxWidth: 280 }}
    >
      {/* Match Quality */}
      <div className="card-sm space-y-2">
        <p className="section-label">Match Quality</p>
        <div className="flex items-end justify-between">
          <span className="text-2xl font-bold tabular-nums" style={{ color: '#23C6A8' }}>
            {Math.round(topScore * 100)}%
          </span>
          <span className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>top result</span>
        </div>
        <div className="score-track">
          <motion.div
            className="score-fill"
            initial={{ width: 0 }}
            animate={{ width: `${Math.round(topScore * 100)}%` }}
            transition={{ duration: 0.8, delay: 0.1 }}
            style={{ background: 'linear-gradient(90deg,#4F8CFF,#23C6A8)' }}
          />
        </div>
        <div className="flex items-center gap-2 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
          <Database className="w-3 h-3" />
          {data.total_found} results · k={data.adaptive_k_used}
          {data.cached && (
            <span className="badge-teal badge ml-auto">cached</span>
          )}
        </div>
      </div>

      {/* Top Resolution */}
      {resolution && (
        <div className="card-sm space-y-2">
          <div className="flex items-center justify-between">
            <p className="section-label">Top Resolution</p>
            <span className="badge badge-teal">{resolution.occurrence_count}× used</span>
          </div>
          <p className="text-[12px] leading-relaxed line-clamp-4"
             style={{ color: 'var(--text-secondary)' }}>
            {resolution.resolution_text}
          </p>
        </div>
      )}

      {/* Category Breakdown */}
      {catEntries.length > 0 && (
        <div className="card-sm space-y-2">
          <p className="section-label">Category Breakdown</p>
          {catEntries.map(([cat, count]) => (
            <div key={cat} className="space-y-0.5">
              <div className="flex justify-between text-[11px]">
                <span style={{ color: 'var(--text-secondary)' }}>{cat}</span>
                <span className="tabular-nums" style={{ color: 'var(--text-primary)' }}>{count}</span>
              </div>
              <div className="score-track" style={{ height: 3 }}>
                <motion.div
                  className="score-fill"
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.round(count / data.total_found * 100)}%` }}
                  transition={{ duration: 0.6, delay: 0.2 }}
                  style={{ background: '#4F8CFF' }}
                />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Triage CTA */}
      <button
        className="btn-primary w-full justify-center"
        style={{ background: 'rgba(79,140,255,0.15)', color: '#4F8CFF',
                 border: '1px solid rgba(79,140,255,0.25)', boxShadow: 'none' }}
        onClick={() => onTriage(data.results[0]?.description ?? '')}
      >
        <ArrowRight className="w-3.5 h-3.5" />
        Triage Top Incident
      </button>
    </motion.div>
  )
}

// ── Main page ─────────────────────────────────────────────
const INITIAL_VISIBLE = 5

export default function SearchPage() {
  const [query,       setQuery]       = useState('')
  const [filters,     setFilters]     = useState<SearchFilters>({})
  const [showFilters, setShowFilters] = useState(false)
  const [showAll,     setShowAll]     = useState(false)

  const navigate = useNavigate()
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const mutation = useMutation({
    mutationFn: () => searchIncidents({ query, filters }),
  })

  const data: SearchResponse | undefined = mutation.data
  const activeFilterCount = Object.values(filters).filter(Boolean).length

  // ── Low-confidence escalation state ──────────────────────────────────────
  const [escalating, setEscalating]     = useState(false)
  const [escalateResult, setEscalateResult] = useState<TriageResult | null>(null)

  const topScore     = data?.results[0]?.similarity_score ?? 1
  const lowConfidence = !!data && !mutation.isPending && topScore < LOW_CONFIDENCE_THRESHOLD

  async function handleEscalate() {
    if (!query.trim() || escalating) return
    setEscalating(true)
    try {
      const result = await runTriage({ description: query, impact: 'Medium', urgency: 'Medium' })
      setEscalateResult(result)
    } catch {
      setEscalateResult(null)
    }
    setEscalating(false)
  }

  // Reset escalation when a new search runs
  function handleSubmitWithReset(e: React.FormEvent) {
    setEscalateResult(null)
    handleSubmit(e)
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (query.trim().length < 3) return
    setShowAll(false)
    mutation.mutate()
  }

  function setExample(s: string) {
    setQuery(s)
    setTimeout(() => inputRef.current?.focus(), 50)
  }

  function toggleFilter(key: keyof SearchFilters, value: string) {
    setFilters(f => ({ ...f, [key]: f[key] === value ? undefined : value }))
  }

  return (
    <div className="flex flex-col gap-5" style={{ minHeight: '100%' }}>
      {/* ── Search header ────────────────────────────────── */}
      <div className="panel" style={{ padding: '20px 24px' }}>
        {/* Controls */}
        <div className="flex items-center justify-end mb-4">
          <button
            className="btn-surface"
            onClick={() => setShowFilters(!showFilters)}
            style={{ color: activeFilterCount > 0 ? 'var(--accent-blue)' : undefined,
                     borderColor: activeFilterCount > 0 ? 'rgba(79,140,255,0.3)' : undefined }}
          >
            <SlidersHorizontal className="w-3.5 h-3.5" />
            Filters
            {activeFilterCount > 0 && (
              <span className="w-4 h-4 rounded-full flex items-center justify-center text-[10px] font-bold text-white"
                    style={{ background: 'var(--accent-blue)' }}>
                {activeFilterCount}
              </span>
            )}
          </button>
        </div>

        {/* Search input */}
        <form onSubmit={handleSubmitWithReset} className="relative">
          <Search className="absolute left-4 top-4 w-5 h-5 pointer-events-none"
                  style={{ color: '#4B5563' }} />
          <textarea
            ref={inputRef}
            className="input-search pr-32"
            placeholder="Describe the incident in natural language — the more specific, the better the results..."
            value={query}
            onChange={e => setQuery(e.target.value)}
            rows={3}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(e as any) } }}
            style={{ resize: 'none', lineHeight: 1.6 }}
          />
          <div className="absolute bottom-3 right-3 flex items-center gap-2">
            {query.length > 0 && (
              <button type="button" className="btn-ghost p-1"
                      onClick={() => setQuery('')}>
                <X className="w-3.5 h-3.5" />
              </button>
            )}
            <button
              type="submit"
              className="btn-primary py-2 px-4"
              disabled={query.trim().length < 3 || mutation.isPending}
            >
              {mutation.isPending ? (
                <span className="flex items-center gap-1.5">
                  <span className="w-3.5 h-3.5 rounded-full border-2 animate-spin"
                        style={{ borderColor: 'rgba(255,255,255,0.3)', borderTopColor: '#fff' }} />
                  Searching
                </span>
              ) : (
                <span className="flex items-center gap-1.5">
                  <Zap className="w-3.5 h-3.5" /> Search
                </span>
              )}
            </button>
          </div>
        </form>

        {/* Filters panel */}
        <AnimatePresence>
          {showFilters && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2 }}
              className="mt-4 pt-4 overflow-hidden"
              style={{ borderTop: '1px solid var(--border)' }}
            >
              <div className="space-y-3">
                {/* Priority */}
                <div className="flex items-center gap-3 flex-wrap">
                  <span className="section-label w-16 flex-shrink-0">Priority</span>
                  {PRIORITY_CHIPS.map(p => (
                    <Chip key={p.value} label={p.label} cls={p.cls}
                          active={filters.priority === p.value}
                          onClick={() => toggleFilter('priority', p.value)} />
                  ))}
                </div>
                {/* Category */}
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="section-label w-16 flex-shrink-0">Category</span>
                  {CATEGORY_CHIPS.map(c => (
                    <Chip key={c} label={c}
                          active={filters.category === c}
                          onClick={() => toggleFilter('category', c)} />
                  ))}
                </div>
                {/* Impact */}
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="section-label w-16 flex-shrink-0">Impact</span>
                  {IMPACT_CHIPS.map(i => (
                    <Chip key={i} label={i}
                          active={filters.impact === i}
                          onClick={() => toggleFilter('impact', i)} />
                  ))}
                </div>
                {activeFilterCount > 0 && (
                  <button className="btn-ghost text-[11px]" style={{ color: 'var(--critical)' }}
                          onClick={() => setFilters({})}>
                    <X className="w-3 h-3" /> Clear all filters
                  </button>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Example queries */}
        {!data && !mutation.isPending && (
          <div className="mt-4 flex flex-wrap gap-2">
            <span className="text-[11px] self-center" style={{ color: 'var(--text-secondary)' }}>
              Try:
            </span>
            {EXAMPLES.map(ex => (
              <button
                key={ex}
                onClick={() => setExample(ex)}
                className="text-[11px] px-2.5 py-1 rounded-lg cursor-pointer transition-all duration-120"
                style={{ background: 'var(--surface)', color: 'var(--text-secondary)',
                         border: '1px solid var(--border)' }}
                onMouseEnter={e => {
                  ;(e.currentTarget as HTMLElement).style.borderColor = 'rgba(79,140,255,0.35)'
                  ;(e.currentTarget as HTMLElement).style.color = 'var(--text-primary)'
                }}
                onMouseLeave={e => {
                  ;(e.currentTarget as HTMLElement).style.borderColor = 'var(--border)'
                  ;(e.currentTarget as HTMLElement).style.color = 'var(--text-secondary)'
                }}
              >
                {ex}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* ── Error ────────────────────────────────────────── */}
      {mutation.isError && (
        <div className="card-sm flex items-center gap-2 text-[13px]"
             style={{ borderColor: 'rgba(240,90,90,0.25)', background: 'rgba(240,90,90,0.06)',
                      color: '#F05A5A' }}>
          <X className="w-4 h-4 flex-shrink-0" />
          {(mutation.error as Error).message}
        </div>
      )}

      {/* ── Loading ───────────────────────────────────────── */}
      {mutation.isPending && (
        <div className="grid grid-cols-1 gap-3">
          {[0, 1, 2].map(i => <SkeletonCard key={i} />)}
        </div>
      )}

      {/* ── Results ───────────────────────────────────────── */}
      {data && !mutation.isPending && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.2 }}
        >
          {/* ── Low confidence banner ──────────────────────── */}
          <AnimatePresence>
            {lowConfidence && (
              <motion.div
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.2 }}
                className="mb-4 rounded-xl p-4"
                style={{
                  background: 'rgba(244,183,64,0.07)',
                  border: '1px solid rgba(244,183,64,0.25)',
                }}
              >
                {/* Not yet escalated */}
                {!escalateResult && (
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5"
                                   style={{ color: '#F4B740' }} />
                    <div className="flex-1 min-w-0">
                      <p className="text-[13px] font-semibold mb-0.5"
                         style={{ color: '#F4B740' }}>
                        Low confidence matches
                      </p>
                      <p className="text-[12px]" style={{ color: 'var(--text-secondary)' }}>
                        The best match is only{' '}
                        <strong style={{ color: 'var(--text-primary)' }}>
                          {Math.round(topScore * 100)}%
                        </strong>{' '}
                        similar — this query may have no relevant historical incidents.
                      </p>
                    </div>
                    <button
                      onClick={handleEscalate}
                      disabled={escalating}
                      className="btn-surface flex-shrink-0"
                      style={{
                        borderColor: 'rgba(244,183,64,0.35)',
                        color: '#F4B740',
                        opacity: escalating ? 0.6 : 1,
                      }}
                    >
                      {escalating ? (
                        <><Loader className="w-3.5 h-3.5 animate-spin" /> Escalating…</>
                      ) : (
                        <><AlertTriangle className="w-3.5 h-3.5" /> Escalate to IT</>
                      )}
                    </button>
                  </div>
                )}

                {/* Escalation result */}
                {escalateResult && (
                  <div className="flex items-start gap-3">
                    {escalateResult.escalation_ticket_id ? (
                      <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5"
                                     style={{ color: '#F05A5A' }} />
                    ) : (
                      <CheckCircle className="w-4 h-4 flex-shrink-0 mt-0.5"
                                   style={{ color: '#23C6A8' }} />
                    )}
                    <div>
                      {escalateResult.escalation_ticket_id ? (
                        <>
                          <p className="text-[13px] font-semibold"
                             style={{ color: '#F05A5A' }}>
                            Escalated to IT team
                          </p>
                          <p className="text-[12px] mt-0.5"
                             style={{ color: 'var(--text-secondary)' }}>
                            Ticket created:{' '}
                            <span className="mono"
                                  style={{ color: 'var(--accent-blue)' }}>
                              {escalateResult.escalation_ticket_id}
                            </span>
                            {' '}— visible in the Admin Portal under Escalation Tickets.
                          </p>
                        </>
                      ) : (
                        <>
                          <p className="text-[13px] font-semibold"
                             style={{ color: '#23C6A8' }}>
                            Resolved by triage agent (
                            {escalateResult.escalation_level})
                          </p>
                          <p className="text-[12px] mt-0.5"
                             style={{ color: 'var(--text-secondary)' }}>
                            {escalateResult.final_answer?.slice(0, 160)}
                            {(escalateResult.final_answer?.length ?? 0) > 160 ? '…' : ''}
                          </p>
                        </>
                      )}
                    </div>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Stats bar */}
          <div className="flex items-center gap-4 mb-4 flex-wrap text-[12px]"
               style={{ color: 'var(--text-secondary)' }}>
            <span className="flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5" style={{ color: '#4F8CFF' }} />
              <strong style={{ color: 'var(--text-primary)' }}>{data.total_found}</strong> relevant incidents
            </span>
            <span className="flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5" style={{ color: '#23C6A8' }} />
              {data.retrieval_method}
            </span>
            <span className="flex items-center gap-1.5 ml-auto">
              <Clock className="w-3.5 h-3.5" />
              {data.latency_ms.toFixed(0)} ms
            </span>
          </div>

          <div className="flex gap-5 items-start">
            {/* Center: results */}
            <div className="flex-1 min-w-0 space-y-4">
              {/* Resolution options */}
              {data.resolution_options.length > 0 && (
                <div className="card" style={{ borderColor: 'rgba(79,140,255,0.2)',
                                               background: 'rgba(79,140,255,0.04)' }}>
                  <div className="flex items-center gap-2 mb-4">
                    <Sparkles className="w-4 h-4" style={{ color: '#4F8CFF' }} />
                    <span className="font-semibold text-[13px]" style={{ color: 'var(--text-primary)' }}>
                      AI-Synthesised Resolution Options
                    </span>
                    <span className="badge badge-blue ml-auto">
                      {data.resolution_options.length} unique fix{data.resolution_options.length !== 1 ? 'es' : ''}
                    </span>
                  </div>
                  <ResolutionPanel options={data.resolution_options} />
                </div>
              )}

              {/* Incident cards */}
              {data.results.length === 0 ? (
                <div className="card text-center py-16">
                  <Search className="w-8 h-8 mx-auto mb-3" style={{ color: 'var(--text-muted)' }} />
                  <p style={{ color: 'var(--text-secondary)' }}>No matching incidents found.</p>
                </div>
              ) : (() => {
                const visible = showAll ? data.results : data.results.slice(0, INITIAL_VISIBLE)
                const hidden  = data.results.length - INITIAL_VISIBLE
                return (
                  <>
                    {visible.map((item, i) => (
                      <motion.div
                        key={item.incident_id}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.04, duration: 0.18 }}
                      >
                        <IncidentCard item={item} rank={i + 1} />
                      </motion.div>
                    ))}

                    {!showAll && hidden > 0 && (
                      <button
                        onClick={() => setShowAll(true)}
                        className="w-full py-2.5 rounded-xl text-[12px] font-semibold transition-all duration-150"
                        style={{
                          background: 'var(--surface)',
                          border: '1px dashed var(--border-strong)',
                          color: 'var(--text-secondary)',
                        }}
                        onMouseEnter={e => {
                          (e.currentTarget as HTMLElement).style.color = '#4F8CFF'
                          ;(e.currentTarget as HTMLElement).style.borderColor = 'rgba(79,140,255,0.4)'
                        }}
                        onMouseLeave={e => {
                          (e.currentTarget as HTMLElement).style.color = 'var(--text-secondary)'
                          ;(e.currentTarget as HTMLElement).style.borderColor = 'var(--border-strong)'
                        }}
                      >
                        Show {hidden} more result{hidden !== 1 ? 's' : ''}
                      </button>
                    )}

                    {showAll && data.results.length > INITIAL_VISIBLE && (
                      <button
                        onClick={() => setShowAll(false)}
                        className="w-full py-2.5 rounded-xl text-[12px] font-semibold transition-all duration-150"
                        style={{
                          background: 'var(--surface)',
                          border: '1px dashed var(--border-strong)',
                          color: 'var(--text-secondary)',
                        }}
                      >
                        Show less
                      </button>
                    )}
                  </>
                )
              })()}
            </div>

            {/* Right: AI Insights */}
            <InsightsPanel
              data={data}
              onTriage={(desc) => navigate('/triage', { state: { description: desc } })}
            />
          </div>
        </motion.div>
      )}

      {/* ── Empty state ───────────────────────────────────── */}
      {!data && !mutation.isPending && !mutation.isError && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="flex flex-col items-center justify-center py-24 text-center"
        >
          <div className="w-16 h-16 rounded-2xl flex items-center justify-center mb-5"
               style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
            <Search className="w-8 h-8" style={{ color: 'var(--text-secondary)' }} />
          </div>
          <h3 className="font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
            Search the incident knowledge base
          </h3>
          <p className="text-[13px] max-w-sm" style={{ color: 'var(--text-secondary)' }}>
            Describe an IT incident in natural language. The system retrieves the most
            similar historical cases and synthesises proven resolutions.
          </p>
          <div className="mt-5 flex items-center gap-4 text-[11px]" style={{ color: '#374151' }}>
            <span>BM25 keyword</span>
            <ChevronRight className="w-3 h-3" />
            <span>Ada-002 semantic</span>
            <ChevronRight className="w-3 h-3" />
            <span>RRF fusion</span>
            <ChevronRight className="w-3 h-3" />
            <span>Cross-encoder rerank</span>
          </div>
        </motion.div>
      )}
    </div>
  )
}
