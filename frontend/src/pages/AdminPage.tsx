import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  ShieldCheck, ThumbsDown, ThumbsUp, Clock, CheckCircle, XCircle,
  RefreshCw, Inbox, MessageSquare, ChevronDown,
} from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts'
import {
  getFeedback, reviewFeedback,
  type FeedbackItem, type FeedbackStatus,
} from '../api/feedbackApi'

// ── Helpers ─────────────────────────────────────────────────────────────────

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

const STATUS_BADGE: Record<FeedbackStatus, string> = {
  PENDING:   'badge-high',
  VERIFIED:  'badge-teal',
  DISMISSED: 'badge-low',
}

const FILTERS: { key: FeedbackStatus | 'ALL'; label: string }[] = [
  { key: 'ALL',       label: 'All' },
  { key: 'PENDING',   label: 'Pending' },
  { key: 'VERIFIED',  label: 'Verified' },
  { key: 'DISMISSED', label: 'Dismissed' },
]

// ── Stat card ───────────────────────────────────────────────────────────────

function StatCard({
  icon: Icon, label, value, color, i,
}: {
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

// ── Feedback row ────────────────────────────────────────────────────────────

function FeedbackRow({
  item, i, expanded, onToggle, onReview, isReviewing,
}: {
  item: FeedbackItem
  i: number
  expanded: boolean
  onToggle: () => void
  onReview: (status: 'VERIFIED' | 'DISMISSED') => void
  isReviewing: boolean
}) {
  const negative = item.sentiment === 'negative'
  const sentimentColor = negative ? '#F05A5A' : '#23C6A8'

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: i * 0.03 }}
      className="card-sm cursor-pointer"
      onClick={onToggle}
    >
      <div className="flex items-start gap-3">
        {/* Sentiment icon */}
        <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5"
             style={{ background: `${sentimentColor}1a` }}>
          {negative
            ? <ThumbsDown className="w-4 h-4" style={{ color: sentimentColor }} />
            : <ThumbsUp className="w-4 h-4" style={{ color: sentimentColor }} />}
        </div>

        {/* Main */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap mb-1">
            <span className={`badge ${STATUS_BADGE[item.status]}`}>{item.status}</span>
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
              “{item.reason}”
            </p>
          )}
        </div>

        <ChevronDown
          className="w-4 h-4 flex-shrink-0 transition-transform"
          style={{
            color: 'var(--text-secondary)',
            transform: expanded ? 'rotate(180deg)' : 'none',
          }}
        />
      </div>

      {/* Expanded detail */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="mt-3 pt-3 overflow-hidden"
            style={{ borderTop: '1px solid var(--border)' }}
            onClick={e => e.stopPropagation()}
          >
            <div className="space-y-2 text-[12px]" style={{ color: 'var(--text-secondary)' }}>
              <Field label="Resolution shown">{item.resolution_text ?? '—'}</Field>
              <Field label="User reason">{item.reason ?? '—'}</Field>
              <div className="flex flex-wrap gap-x-6 gap-y-1">
                <Field label="Verified in KB">{item.occurrence_count}× historically</Field>
                <Field label="Source incidents">
                  {item.incident_ids.length ? item.incident_ids.join(', ') : '—'}
                </Field>
                <Field label="Session">
                  <span className="mono">{item.session_id ?? '—'}</span>
                </Field>
              </div>
              {item.admin_action && <Field label="Admin note">{item.admin_action}</Field>}
            </div>

            {/* Actions */}
            <div className="flex items-center gap-2 mt-3">
              <button
                className="btn-primary"
                disabled={isReviewing || item.status === 'VERIFIED'}
                onClick={() => onReview('VERIFIED')}
                style={{ background: '#23C6A8' }}
              >
                <CheckCircle className="w-4 h-4" /> Verify
              </button>
              <button
                className="btn-surface"
                disabled={isReviewing || item.status === 'DISMISSED'}
                onClick={() => onReview('DISMISSED')}
              >
                <XCircle className="w-4 h-4" /> Dismiss
              </button>
              {isReviewing && (
                <RefreshCw className="w-4 h-4 animate-spin" style={{ color: 'var(--accent-blue)' }} />
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
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

const ChartTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="card-sm text-[12px]" style={{ minWidth: 110 }}>
      <p className="font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>{label}</p>
      <p style={{ color: '#F05A5A' }}>{payload[0]?.value} negative</p>
    </div>
  )
}

// ── Page ────────────────────────────────────────────────────────────────────

export default function AdminPage() {
  const [filter, setFilter] = useState<FeedbackStatus | 'ALL'>('ALL')
  const [expanded, setExpanded] = useState<string | null>(null)
  const qc = useQueryClient()

  const { data, isFetching, refetch } = useQuery({
    queryKey: ['feedback', filter],
    queryFn: () => getFeedback(filter === 'ALL' ? undefined : filter),
    refetchInterval: 4000,   // live: new chat feedback appears within ~4s
  })

  const reviewMutation = useMutation({
    mutationFn: ({ id, status }: { id: string; status: 'VERIFIED' | 'DISMISSED' }) =>
      reviewFeedback(id, status),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['feedback'] }),
  })

  const stats = data?.stats
  const items = data?.items ?? []

  // Negative feedback grouped by fix index → bar chart
  const byFix = items
    .filter(it => it.sentiment === 'negative')
    .reduce<Record<number, number>>((acc, it) => {
      acc[it.fix_index] = (acc[it.fix_index] ?? 0) + 1
      return acc
    }, {})
  const chartData = Object.entries(byFix)
    .map(([fix, count]) => ({ name: `Fix ${fix}`, count }))
    .sort((a, b) => a.name.localeCompare(b.name))

  return (
    <div className="space-y-5">
      {/* ── Header ─────────────────────────────────────── */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-5 h-5" style={{ color: 'var(--accent-blue)' }} />
          <h1 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
            Feedback Review
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

      {/* ── Stat cards ─────────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard icon={Inbox}     label="Total feedback" value={stats?.total ?? 0}    color="#4F8CFF" i={0} />
        <StatCard icon={ThumbsDown} label="Negative"       value={stats?.negative ?? 0} color="#F05A5A" i={1} />
        <StatCard icon={Clock}     label="Pending review"  value={stats?.pending ?? 0}  color="#F4B740" i={2} />
        <StatCard icon={CheckCircle} label="Verified"      value={stats?.verified ?? 0} color="#23C6A8" i={3} />
      </div>

      {/* ── Chart + intro ──────────────────────────────── */}
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
                {chartData.map((_, idx) => (
                  <Cell key={idx} fill="#F05A5A" fillOpacity={0.8} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ── Filter tabs ────────────────────────────────── */}
      <div className="flex items-center gap-2 flex-wrap">
        {FILTERS.map(f => {
          const active = filter === f.key
          return (
            <button
              key={f.key}
              onClick={() => setFilter(f.key)}
              className="text-[12px] px-3 py-1.5 rounded-lg font-medium transition-colors"
              style={{
                background: active ? 'rgba(79,140,255,0.12)' : 'var(--surface)',
                border: `1px solid ${active ? 'rgba(79,140,255,0.25)' : 'var(--border)'}`,
                color: active ? 'var(--accent-blue)' : 'var(--text-secondary)',
              }}
            >
              {f.label}
            </button>
          )
        })}
      </div>

      {/* ── Feedback list ──────────────────────────────── */}
      {items.length === 0 ? (
        <div className="card text-center py-20"
             style={{ borderStyle: 'dashed', borderColor: 'rgba(255,255,255,0.08)' }}>
          <div className="w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-4"
               style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
            <MessageSquare className="w-7 h-7" style={{ color: 'var(--text-secondary)' }} />
          </div>
          <p className="font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>
            No feedback to review
          </p>
          <p className="text-[12px]" style={{ color: 'var(--text-secondary)' }}>
            When a user reacts to a fix in the chat assistant, it appears here in real time.
          </p>
        </div>
      ) : (
        <div className="space-y-2.5">
          <AnimatePresence initial={false}>
            {items.map((item, i) => (
              <FeedbackRow
                key={item.feedback_id}
                item={item}
                i={i}
                expanded={expanded === item.feedback_id}
                onToggle={() =>
                  setExpanded(prev => (prev === item.feedback_id ? null : item.feedback_id))}
                onReview={status => reviewMutation.mutate({ id: item.feedback_id, status })}
                isReviewing={
                  reviewMutation.isPending &&
                  reviewMutation.variables?.id === item.feedback_id
                }
              />
            ))}
          </AnimatePresence>
        </div>
      )}
    </div>
  )
}
