import { useState, useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  ShieldCheck, ThumbsDown, ThumbsUp, Clock, CheckCircle, XCircle,
  RefreshCw, Inbox, MessageSquare, ChevronDown, AlertTriangle, Ticket,
} from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts'
import {
  getFeedback, reviewFeedback,
  type FeedbackItem, type FeedbackStatus,
} from '../api/feedbackApi'
import { getEscalations, resolveEscalation, type EscalationTicket } from '../api/triageApi'

// ── Helpers ──────────────────────────────────────────────────────────────────

function relativeTime(iso: string | null): string {
  if (!iso) return '—'
  const then = new Date(iso).getTime()
  const diff = Date.now() - then
  const mins = Math.round(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins} min ago`
  const hrs = Math.round(mins / 60)
  if (hrs < 24) return `${hrs} hr${hrs === 1 ? '' : 's'} ago`
  const days = Math.round(hrs / 24)
  return `${days} day${days === 1 ? '' : 's'} ago`
}

// ── Stat card ────────────────────────────────────────────────────────────────

function StatCard({ icon: Icon, label, value, color, i }: {
  icon: typeof Inbox; label: string; value: number; color: string; i: number
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: i * 0.05 }}
      className="stat-card"
      style={{ borderColor: `${color}33` }}
    >
      <div className="absolute top-0 left-0 right-0 h-0.5 rounded-t-xl"
           style={{ background: color, opacity: 0.7 }} />
      <div className="w-8 h-8 rounded-xl flex items-center justify-center mb-2"
           style={{ background: `${color}1a` }}>
        <Icon className="w-4 h-4" style={{ color }} />
      </div>
      <p className="text-[26px] font-bold tabular-nums" style={{ color: 'var(--text-primary)' }}>
        {value}
      </p>
      <p className="text-[11px] font-semibold" style={{ color: 'var(--text-secondary)' }}>
        {label}
      </p>
    </motion.div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>{label}: </span>
      <span>{children}</span>
    </div>
  )
}

// ── Feedback components ───────────────────────────────────────────────────────

const FEEDBACK_STATUS_BADGE: Record<FeedbackStatus, string> = {
  PENDING:   'badge-high',
  VERIFIED:  'badge-teal',
  DISMISSED: 'badge-low',
}

const FEEDBACK_FILTERS: { key: FeedbackStatus | 'ALL'; label: string }[] = [
  { key: 'ALL',       label: 'All' },
  { key: 'PENDING',   label: 'Pending' },
  { key: 'VERIFIED',  label: 'Verified' },
  { key: 'DISMISSED', label: 'Dismissed' },
]

const ChartTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="card-sm text-[12px]" style={{ minWidth: 110 }}>
      <p className="font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>{label}</p>
      <p style={{ color: '#F05A5A' }}>{payload[0]?.value} negative</p>
    </div>
  )
}

function FeedbackRow({ item, i, expanded, onToggle, onReview, isReviewing }: {
  item: FeedbackItem; i: number; expanded: boolean
  onToggle: () => void; onReview: (s: 'VERIFIED' | 'DISMISSED') => void; isReviewing: boolean
}) {
  const negative = item.sentiment === 'negative'
  const color = negative ? '#F05A5A' : '#23C6A8'

  return (
    <motion.div layout initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
      transition={{ delay: i * 0.03 }} className="card-sm cursor-pointer" onClick={onToggle}>
      <div className="flex items-start gap-3">
        <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5"
             style={{ background: `${color}1a` }}>
          {negative ? <ThumbsDown className="w-4 h-4" style={{ color }} />
                    : <ThumbsUp className="w-4 h-4" style={{ color }} />}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap mb-1">
            <span className={`badge ${FEEDBACK_STATUS_BADGE[item.status]}`}>{item.status}</span>
            <span className="mono px-1.5 py-0.5 rounded-md"
                  style={{ background: 'rgba(255,255,255,0.06)', color: 'var(--text-secondary)' }}>
              Fix {item.fix_index}/{item.fix_total}
            </span>
            <span className="flex items-center gap-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
              <Clock className="w-3 h-3" /> {relativeTime(item.created_at)}
            </span>
          </div>
          <p className="text-[13px] font-medium truncate" style={{ color: 'var(--text-primary)' }}>
            {item.query}
          </p>
          {item.reason && (
            <p className="text-[12px] mt-0.5 truncate" style={{ color: 'var(--text-secondary)' }}>
              "{item.reason}"
            </p>
          )}
        </div>
        <ChevronDown className="w-4 h-4 flex-shrink-0 transition-transform"
          style={{ color: 'var(--text-secondary)', transform: expanded ? 'rotate(180deg)' : 'none' }} />
      </div>

      <AnimatePresence>
        {expanded && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }} transition={{ duration: 0.2 }}
            className="mt-3 pt-3 overflow-hidden" style={{ borderTop: '1px solid var(--border)' }}
            onClick={e => e.stopPropagation()}>
            <div className="space-y-2 text-[12px]" style={{ color: 'var(--text-secondary)' }}>
              <Field label="Resolution shown">{item.resolution_text ?? '—'}</Field>
              <Field label="User reason">{item.reason ?? '—'}</Field>
              <div className="flex flex-wrap gap-x-6 gap-y-1">
                <Field label="Verified in KB">{item.occurrence_count}× historically</Field>
                <Field label="Source incidents">
                  {item.incident_ids.length ? item.incident_ids.join(', ') : '—'}
                </Field>
                <Field label="Session"><span className="mono">{item.session_id ?? '—'}</span></Field>
              </div>
              {item.admin_action && <Field label="Admin note">{item.admin_action}</Field>}
            </div>
            <div className="flex items-center gap-2 mt-3">
              <button className="btn-primary" disabled={isReviewing || item.status === 'VERIFIED'}
                onClick={() => onReview('VERIFIED')} style={{ background: '#23C6A8' }}>
                <CheckCircle className="w-4 h-4" /> Verify
              </button>
              <button className="btn-surface" disabled={isReviewing || item.status === 'DISMISSED'}
                onClick={() => onReview('DISMISSED')}>
                <XCircle className="w-4 h-4" /> Dismiss
              </button>
              {isReviewing && <RefreshCw className="w-4 h-4 animate-spin" style={{ color: 'var(--accent-blue)' }} />}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

// ── Escalation ticket components ──────────────────────────────────────────────

const TICKET_STATUS_BADGE: Record<string, { cls: string; color: string }> = {
  OPEN:        { cls: 'badge-critical', color: '#F05A5A' },
  IN_PROGRESS: { cls: 'badge-high',     color: '#F4B740' },
  RESOLVED:    { cls: 'badge-teal',     color: '#23C6A8' },
}

const TICKET_FILTERS = ['ALL', 'OPEN', 'IN_PROGRESS', 'RESOLVED']

function TicketRow({ ticket, i, expanded, onToggle, onResolved }: {
  ticket: EscalationTicket; i: number; expanded: boolean
  onToggle: () => void; onResolved: () => void
}) {
  const badge = TICKET_STATUS_BADGE[ticket.status] ?? TICKET_STATUS_BADGE.OPEN
  const [resolveOpen, setResolveOpen]   = useState(false)
  const [steps, setSteps]               = useState('')
  const [submitting, setSubmitting]     = useState(false)
  const [resolvedMsg, setResolvedMsg]   = useState('')

  async function handleResolve() {
    if (!steps.trim() || submitting) return
    setSubmitting(true)
    try {
      const res = await resolveEscalation(ticket.ticket_id, steps.trim())
      setResolvedMsg(res.ingested_to_kb
        ? `Resolved. Resolution added to KB as ${res.new_incident_id}.`
        : 'Marked resolved. KB ingestion failed — check embedding service.')
      setResolveOpen(false)
      setTimeout(onResolved, 1200)   // refresh after brief delay so user sees the message
    } catch (e: any) {
      setResolvedMsg(`Error: ${e.message}`)
    }
    setSubmitting(false)
  }

  return (
    <motion.div layout initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
      transition={{ delay: i * 0.03 }} className="card-sm">
      {/* ── Header row — always visible ───────────────────────── */}
      <div className="flex items-start gap-3 cursor-pointer" onClick={onToggle}>
        <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5"
             style={{ background: `${badge.color}1a` }}>
          <AlertTriangle className="w-4 h-4" style={{ color: badge.color }} />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap mb-1">
            <span className="mono text-[11px] px-1.5 py-0.5 rounded-md"
                  style={{ background: 'rgba(255,255,255,0.06)', color: 'var(--accent-blue)' }}>
              {ticket.ticket_id}
            </span>
            <span className={`badge ${badge.cls}`}>{ticket.status.replace('_', ' ')}</span>
            {ticket.urgency && <span className="badge badge-medium">{ticket.urgency} urgency</span>}
            <span className="flex items-center gap-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
              <Clock className="w-3 h-3" /> {relativeTime(ticket.created_at)}
            </span>
          </div>
          <p className="text-[13px] font-medium truncate" style={{ color: 'var(--text-primary)' }}>
            {ticket.description}
          </p>
          {ticket.escalation_reason && (
            <p className="text-[12px] mt-0.5 truncate" style={{ color: 'var(--text-secondary)' }}>
              {ticket.escalation_reason}
            </p>
          )}
        </div>

        <ChevronDown className="w-4 h-4 flex-shrink-0 transition-transform"
          style={{ color: 'var(--text-secondary)', transform: expanded ? 'rotate(180deg)' : 'none' }} />
      </div>

      {/* ── Expanded detail ───────────────────────────────────── */}
      <AnimatePresence>
        {expanded && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }} transition={{ duration: 0.2 }}
            className="mt-3 pt-3 overflow-hidden" style={{ borderTop: '1px solid var(--border)' }}>

            <div className="space-y-2.5 text-[12px]" style={{ color: 'var(--text-secondary)' }}>
              <Field label="Escalation reason">{ticket.escalation_reason || '—'}</Field>
              {ticket.l1_summary && (
                <div>
                  <p className="font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>L1 Summary</p>
                  <p className="leading-relaxed p-2 rounded-lg"
                     style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border)' }}>
                    {ticket.l1_summary}
                  </p>
                </div>
              )}
              {ticket.l2_analysis && (
                <div>
                  <p className="font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>L2 Analysis</p>
                  <p className="leading-relaxed p-2 rounded-lg"
                     style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border)' }}>
                    {ticket.l2_analysis}
                  </p>
                </div>
              )}
              <div className="flex flex-wrap gap-x-6 gap-y-1">
                {ticket.impact  && <Field label="Impact">{ticket.impact}</Field>}
                {ticket.urgency && <Field label="Urgency">{ticket.urgency}</Field>}
              </div>
            </div>

            {/* ── Resolve action ────────────────────────────────── */}
            {ticket.status !== 'RESOLVED' && !resolvedMsg && (
              <div className="mt-3">
                {!resolveOpen ? (
                  <button
                    className="btn-primary"
                    onClick={e => { e.stopPropagation(); setResolveOpen(true) }}
                    style={{ background: '#23C6A8' }}
                  >
                    <CheckCircle className="w-4 h-4" /> Mark as Resolved
                  </button>
                ) : (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    className="overflow-hidden"
                    onClick={e => e.stopPropagation()}
                  >
                    <p className="text-[11px] mb-2" style={{ color: 'var(--text-secondary)' }}>
                      What steps did you take to resolve this?
                      <span className="ml-1 opacity-50">(will be added to the knowledge base)</span>
                    </p>
                    <textarea
                      autoFocus
                      value={steps}
                      onChange={e => setSteps(e.target.value)}
                      placeholder="e.g. Replaced the failed NIC card, updated drivers, tested connectivity..."
                      rows={4}
                      className="w-full resize-none rounded-xl px-3 py-2.5 text-[12px] focus:outline-none"
                      style={{
                        background: 'rgba(255,255,255,0.05)',
                        border: '1px solid var(--border)',
                        color: 'var(--text-primary)',
                      }}
                      onFocus={e => (e.currentTarget.style.borderColor = 'rgba(35,198,168,0.4)')}
                      onBlur={e => (e.currentTarget.style.borderColor = 'var(--border)')}
                    />
                    <div className="flex items-center gap-2 mt-2">
                      <button
                        onClick={handleResolve}
                        disabled={submitting || steps.trim().length < 10}
                        className="btn-primary disabled:opacity-40"
                        style={{ background: '#23C6A8' }}
                      >
                        <CheckCircle className="w-4 h-4" />
                        {submitting ? 'Saving…' : 'Confirm & Add to KB'}
                      </button>
                      <button
                        onClick={() => { setResolveOpen(false); setSteps('') }}
                        className="btn-ghost text-[11px]"
                      >
                        Cancel
                      </button>
                    </div>
                  </motion.div>
                )}
              </div>
            )}

            {/* Resolved confirmation */}
            {resolvedMsg && (
              <div className="mt-3 flex items-center gap-2 text-[11px] px-3 py-2 rounded-lg"
                   style={{ background: 'rgba(35,198,168,0.1)', color: '#23C6A8',
                            border: '1px solid rgba(35,198,168,0.2)' }}>
                <CheckCircle className="w-3.5 h-3.5 flex-shrink-0" />
                {resolvedMsg}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

// ── Page ─────────────────────────────────────────────────────────────────────

type Tab = 'feedback' | 'escalations'

export default function AdminPage() {
  const [activeTab, setActiveTab] = useState<Tab>('feedback')

  // ── Feedback state ────────────────────────────────────────────────────────
  const [feedbackFilter, setFeedbackFilter] = useState<FeedbackStatus | 'ALL'>('ALL')
  const [expandedFeedback, setExpandedFeedback] = useState<string | null>(null)
  const qc = useQueryClient()

  const { data: feedbackData, isFetching: feedbackFetching, refetch: refetchFeedback } = useQuery({
    queryKey: ['feedback', feedbackFilter],
    queryFn: () => getFeedback(feedbackFilter === 'ALL' ? undefined : feedbackFilter),
    refetchInterval: 4000,
  })

  const reviewMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: 'VERIFIED' | 'DISMISSED' }) =>
      reviewFeedback(id, status),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['feedback'] }),
  })

  // ── Escalations state ─────────────────────────────────────────────────────
  const [ticketFilter, setTicketFilter] = useState('ALL')
  const [expandedTicket, setExpandedTicket] = useState<string | null>(null)

  const { data: escalationData, isFetching: ticketFetching, refetch: refetchTickets } = useQuery({
    queryKey: ['escalations', ticketFilter],
    queryFn: () => getEscalations(ticketFilter === 'ALL' ? undefined : ticketFilter, 200),
    refetchInterval: 10000,
  })

  const tickets = escalationData?.tickets ?? []
  const ticketStats = {
    total:       tickets.length,
    open:        tickets.filter(t => t.status === 'OPEN').length,
    in_progress: tickets.filter(t => t.status === 'IN_PROGRESS').length,
    resolved:    tickets.filter(t => t.status === 'RESOLVED').length,
  }

  // ── Feedback derived values ───────────────────────────────────────────────
  const stats = feedbackData?.stats
  const feedbackItems = feedbackData?.items ?? []
  const byFix = feedbackItems
    .filter(it => it.sentiment === 'negative')
    .reduce<Record<number, number>>((acc, it) => {
      acc[it.fix_index] = (acc[it.fix_index] ?? 0) + 1; return acc
    }, {})
  const chartData = Object.entries(byFix)
    .map(([fix, count]) => ({ name: `Fix ${fix}`, count }))
    .sort((a, b) => a.name.localeCompare(b.name))

  const isFetching = activeTab === 'feedback' ? feedbackFetching : ticketFetching
  const refetch    = activeTab === 'feedback' ? refetchFeedback   : refetchTickets

  return (
    <div className="space-y-5">

      {/* ── Header ───────────────────────────────────────── */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-5 h-5" style={{ color: 'var(--accent-blue)' }} />
          <h1 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
            Admin Console
          </h1>
          <span className="text-[11px] px-2 py-0.5 rounded-full font-mono"
                style={{ background: 'rgba(79,140,255,0.12)', color: 'var(--accent-blue)' }}>
            Admin
          </span>
        </div>
        <div className="flex items-center gap-2 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
          <span className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full pulse-dot" style={{ background: 'var(--accent-teal)' }} />
            Live · auto-refreshing
          </span>
          <button className="btn-surface" onClick={() => refetch()} disabled={isFetching}>
            <RefreshCw className={`w-3.5 h-3.5 ${isFetching ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* ── Tab switcher ─────────────────────────────────── */}
      <div className="flex gap-1 p-1 rounded-xl w-fit"
           style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
        {([
          { key: 'feedback',    label: 'Feedback Review',     icon: ThumbsDown,
            count: stats?.pending ?? 0, countColor: '#F05A5A' },
          { key: 'escalations', label: 'Escalation Tickets',  icon: Ticket,
            count: ticketStats.open,    countColor: '#F05A5A' },
        ] as const).map(tab => {
          const active = activeTab === tab.key
          return (
            <button key={tab.key} onClick={() => setActiveTab(tab.key)}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-[12px] font-medium transition-all"
              style={{
                background: active ? 'var(--card)' : 'transparent',
                color: active ? 'var(--text-primary)' : 'var(--text-secondary)',
                border: active ? '1px solid var(--border-strong)' : '1px solid transparent',
              }}>
              <tab.icon className="w-3.5 h-3.5" />
              {tab.label}
              {tab.count > 0 && (
                <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-full min-w-[18px] text-center"
                      style={{ background: `${tab.countColor}1a`, color: tab.countColor,
                               border: `1px solid ${tab.countColor}33` }}>
                  {tab.count}
                </span>
              )}
            </button>
          )
        })}
      </div>

      {/* ══════════════════════════════════════════════════ */}
      {/* FEEDBACK TAB                                       */}
      {/* ══════════════════════════════════════════════════ */}
      {activeTab === 'feedback' && (
        <motion.div key="feedback" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-5">

          {/* Stat cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <StatCard icon={Inbox}       label="Total feedback"  value={stats?.total   ?? 0} color="#4F8CFF" i={0} />
            <StatCard icon={ThumbsDown}  label="Negative"        value={stats?.negative ?? 0} color="#F05A5A" i={1} />
            <StatCard icon={Clock}       label="Pending review"  value={stats?.pending  ?? 0} color="#F4B740" i={2} />
            <StatCard icon={CheckCircle} label="Verified"        value={stats?.verified ?? 0} color="#23C6A8" i={3} />
          </div>

          {/* Chart */}
          {chartData.length > 0 && (
            <div className="card">
              <div className="flex items-center gap-2 mb-4">
                <ThumbsDown className="w-4 h-4" style={{ color: '#F05A5A' }} />
                <p className="font-semibold text-[13px]" style={{ color: 'var(--text-primary)' }}>
                  Negative feedback by fix position
                </p>
              </div>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={chartData} margin={{ top: 4, right: 8, left: -24, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.04)" />
                  <XAxis dataKey="name" tick={{ fontSize: 11, fill: 'var(--text-secondary)' }} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 10, fill: '#4B5563' }} />
                  <Tooltip content={<ChartTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
                  <Bar dataKey="count" radius={[6, 6, 0, 0]} maxBarSize={48}>
                    {chartData.map((_, idx) => <Cell key={idx} fill="#F05A5A" fillOpacity={0.8} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Filter tabs */}
          <div className="flex items-center gap-2 flex-wrap">
            {FEEDBACK_FILTERS.map(f => {
              const active = feedbackFilter === f.key
              return (
                <button key={f.key} onClick={() => setFeedbackFilter(f.key)}
                  className="text-[12px] px-3 py-1.5 rounded-lg font-medium transition-colors"
                  style={{
                    background: active ? 'rgba(79,140,255,0.12)' : 'var(--surface)',
                    border: `1px solid ${active ? 'rgba(79,140,255,0.25)' : 'var(--border)'}`,
                    color: active ? 'var(--accent-blue)' : 'var(--text-secondary)',
                  }}>
                  {f.label}
                </button>
              )
            })}
          </div>

          {/* Feedback list */}
          {feedbackItems.length === 0 ? (
            <div className="card text-center py-20"
                 style={{ borderStyle: 'dashed', borderColor: 'rgba(255,255,255,0.08)' }}>
              <div className="w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-4"
                   style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
                <MessageSquare className="w-7 h-7" style={{ color: 'var(--text-secondary)' }} />
              </div>
              <p className="font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>No feedback to review</p>
              <p className="text-[12px]" style={{ color: 'var(--text-secondary)' }}>
                When a user rates a fix in the chat assistant, it appears here.
              </p>
            </div>
          ) : (
            <div className="space-y-2.5">
              <AnimatePresence initial={false}>
                {feedbackItems.map((item, i) => (
                  <FeedbackRow key={item.feedback_id} item={item} i={i}
                    expanded={expandedFeedback === item.feedback_id}
                    onToggle={() => setExpandedFeedback(p => p === item.feedback_id ? null : item.feedback_id)}
                    onReview={status => reviewMutation.mutate({ id: item.feedback_id, status })}
                    isReviewing={reviewMutation.isPending && reviewMutation.variables?.id === item.feedback_id}
                  />
                ))}
              </AnimatePresence>
            </div>
          )}
        </motion.div>
      )}

      {/* ══════════════════════════════════════════════════ */}
      {/* ESCALATION TICKETS TAB                             */}
      {/* ══════════════════════════════════════════════════ */}
      {activeTab === 'escalations' && (
        <motion.div key="escalations" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-5">

          {/* Stat cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <StatCard icon={Ticket}       label="Total tickets"  value={ticketStats.total}       color="#4F8CFF" i={0} />
            <StatCard icon={AlertTriangle} label="Open"          value={ticketStats.open}        color="#F05A5A" i={1} />
            <StatCard icon={RefreshCw}    label="In Progress"    value={ticketStats.in_progress} color="#F4B740" i={2} />
            <StatCard icon={CheckCircle}  label="Resolved"       value={ticketStats.resolved}    color="#23C6A8" i={3} />
          </div>

          {/* Filter tabs */}
          <div className="flex items-center gap-2 flex-wrap">
            {TICKET_FILTERS.map(f => {
              const active = ticketFilter === f
              return (
                <button key={f} onClick={() => setTicketFilter(f)}
                  className="text-[12px] px-3 py-1.5 rounded-lg font-medium transition-colors"
                  style={{
                    background: active ? 'rgba(240,90,90,0.12)' : 'var(--surface)',
                    border: `1px solid ${active ? 'rgba(240,90,90,0.25)' : 'var(--border)'}`,
                    color: active ? '#F05A5A' : 'var(--text-secondary)',
                  }}>
                  {f.replace('_', ' ')}
                </button>
              )
            })}
          </div>

          {/* Ticket list */}
          {tickets.length === 0 ? (
            <div className="card text-center py-20"
                 style={{ borderStyle: 'dashed', borderColor: 'rgba(255,255,255,0.08)' }}>
              <div className="w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-4"
                   style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
                <Ticket className="w-7 h-7" style={{ color: 'var(--text-secondary)' }} />
              </div>
              <p className="font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>No escalation tickets</p>
              <p className="text-[12px]" style={{ color: 'var(--text-secondary)' }}>
                Tickets appear here when the L3 triage agent or chat assistant escalates an incident.
              </p>
            </div>
          ) : (
            <div className="space-y-2.5">
              <AnimatePresence initial={false}>
                {tickets.map((ticket, i) => (
                  <TicketRow key={ticket.ticket_id} ticket={ticket} i={i}
                    expanded={expandedTicket === ticket.ticket_id}
                    onToggle={() => setExpandedTicket(p => p === ticket.ticket_id ? null : ticket.ticket_id)}
                    onResolved={() => { setExpandedTicket(null); refetchTickets() }}
                  />
                ))}
              </AnimatePresence>
            </div>
          )}
        </motion.div>
      )}
    </div>
  )
}
