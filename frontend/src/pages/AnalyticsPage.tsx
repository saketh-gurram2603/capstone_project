import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import {
  BarChart2, CheckCircle, XCircle, Play, RefreshCw,
  TrendingUp, Target, Brain, Zap,
} from 'lucide-react'
import { getLatestMetrics, runEvaluation, type MetricScore } from '../api/evaluationApi'
import MetricChart from '../components/MetricChart'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts'

const LABEL_MAP: Record<string, string> = {
  ndcg_at_k:                'NDCG@10',
  map_at_k:                 'MAP@10',
  recall_at_k:              'Recall@10',
  precision_at_k:           'P@10',
  avg_faithfulness:         'Faithfulness',
  avg_answer_relevancy:     'Relevancy',
  avg_contextual_precision: 'Ctx Precision',
}

const ICONS: Record<string, typeof TrendingUp> = {
  ndcg_at_k:                TrendingUp,
  map_at_k:                 Target,
  recall_at_k:              BarChart2,
  precision_at_k:           BarChart2,
  avg_faithfulness:         Brain,
  avg_answer_relevancy:     Brain,
  avg_contextual_precision: Brain,
}

const DESC: Record<string, string> = {
  ndcg_at_k:                'Ranking quality',
  map_at_k:                 'Avg precision',
  recall_at_k:              'Coverage rate',
  precision_at_k:           'Result accuracy',
  avg_faithfulness:         'LLM-as-Judge',
  avg_answer_relevancy:     'LLM-as-Judge',
  avg_contextual_precision: 'LLM-as-Judge',
}

function MetricCard({
  m, i, expanded, onToggle,
}: {
  m: MetricScore; i: number; expanded: boolean; onToggle: () => void
}) {
  const pct   = Math.round(m.score * 100)
  const tPct  = Math.round(m.threshold * 100)
  const Icon  = ICONS[m.name] ?? TrendingUp
  const color = m.passed ? '#23C6A8' : '#F05A5A'
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: i * 0.06 }}
      className="stat-card cursor-pointer select-none"
      style={{ borderColor: m.passed ? 'rgba(35,198,168,0.2)' : 'rgba(240,90,90,0.2)' }}
      onClick={onToggle}
      title={expanded ? 'Click to collapse' : 'Click to see why'}
    >
      {/* Top glow strip */}
      <div className="absolute top-0 left-0 right-0 h-0.5 rounded-t-xl"
           style={{ background: color, opacity: 0.7 }} />

      <div className="flex items-start justify-between mb-3">
        <div className="w-8 h-8 rounded-xl flex items-center justify-center"
             style={{ background: m.passed ? 'rgba(35,198,168,0.1)' : 'rgba(240,90,90,0.1)' }}>
          <Icon className="w-4 h-4" style={{ color }} />
        </div>
        {m.passed
          ? <CheckCircle className="w-4 h-4" style={{ color: '#23C6A8' }} />
          : <XCircle    className="w-4 h-4" style={{ color: '#F05A5A' }} />
        }
      </div>

      <p className="text-[26px] font-bold tabular-nums" style={{ color: 'var(--text-primary)' }}>
        {pct}<span className="text-[16px]">%</span>
      </p>
      <p className="text-[11px] font-semibold" style={{ color: 'var(--text-primary)' }}>
        {LABEL_MAP[m.name] ?? m.name}
      </p>
      <p className="text-[10px] mb-3" style={{ color: 'var(--text-secondary)' }}>
        {DESC[m.name] ?? ''}
      </p>

      <div className="score-track">
        <motion.div
          className="score-fill"
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8, delay: i * 0.06 + 0.1 }}
          style={{ background: color }}
        />
      </div>
      <p className="text-[10px] mt-1.5" style={{ color: '#374151' }}>
        Threshold: {tPct}%
      </p>

      {/* ── "Why this score" expandable panel ─────────────────────────── */}
      {expanded && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.2 }}
          className="mt-3 pt-3 text-[11px]"
          style={{ borderTop: '1px solid var(--border)' }}
          onClick={e => e.stopPropagation()}
        >
          <p className="font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>
            Why this score
          </p>
          {m.reason ? (
            <p style={{ color: 'var(--text-secondary)', lineHeight: '1.5' }}>{m.reason}</p>
          ) : (
            <p style={{ color: '#4B5563', fontStyle: 'italic' }}>
              Run evaluation to generate an explanation.
            </p>
          )}
        </motion.div>
      )}
    </motion.div>
  )
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="card-sm text-[12px]" style={{ minWidth: 120 }}>
      <p className="font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>{label}</p>
      <p style={{ color: payload[0]?.payload?.passed ? '#23C6A8' : '#F05A5A' }}>
        Score: {payload[0]?.value}%
      </p>
    </div>
  )
}

export default function AnalyticsPage() {
  const [runLlmJudge,   setRunLlmJudge]   = useState(false)
  const [expandedCard,  setExpandedCard]   = useState<string | null>(null)

  const { data: latest, refetch, isFetching } = useQuery({
    queryKey: ['metrics'],
    queryFn: getLatestMetrics,
  })

  const evalMutation = useMutation({
    mutationFn: () => runEvaluation({ run_ir_metrics: true, run_llm_judge: runLlmJudge }),
    onSuccess: () => refetch(),
  })

  const metrics  = latest?.metrics ?? []
  const hasData  = metrics.length > 0
  const passed   = metrics.filter(m => m.passed).length

  const barData = metrics.map(m => ({
    name:      LABEL_MAP[m.name] ?? m.name,
    score:     Math.round(m.score * 100),
    threshold: Math.round(m.threshold * 100),
    passed:    m.passed,
  }))

  return (
    <div className="space-y-5">
      {/* ── Controls ─────────────────────────────────────── */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 cursor-pointer btn-surface"
                 style={{ userSelect: 'none' }}>
            <input
              type="checkbox"
              checked={runLlmJudge}
              onChange={e => setRunLlmJudge(e.target.checked)}
              className="rounded"
              style={{ accentColor: 'var(--accent-blue)' }}
            />
            <Brain className="w-3.5 h-3.5" />
            <span className="text-[12px]">LLM Judge</span>
          </label>
        </div>
        <div className="flex items-center gap-2">
          {hasData && (
            <button className="btn-surface" onClick={() => refetch()} disabled={isFetching}>
              <RefreshCw className={`w-3.5 h-3.5 ${isFetching ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          )}
          <button
            className="btn-primary"
            onClick={() => evalMutation.mutate()}
            disabled={evalMutation.isPending}
          >
            {evalMutation.isPending
              ? <><RefreshCw className="w-4 h-4 animate-spin" /> Running…</>
              : <><Play className="w-4 h-4" /> Run Evaluation</>
            }
          </button>
        </div>
      </div>

      {/* Error */}
      {evalMutation.isError && (
        <div className="card-sm text-[12px] flex items-center gap-2"
             style={{ color: '#F05A5A', borderColor: 'rgba(240,90,90,0.2)',
                      background: 'rgba(240,90,90,0.05)' }}>
          {(evalMutation.error as Error).message}
        </div>
      )}

      {/* Running state */}
      {evalMutation.isPending && (
        <div className="card text-center py-16">
          <RefreshCw className="w-7 h-7 animate-spin mx-auto mb-3" style={{ color: '#4F8CFF' }} />
          <p className="font-medium" style={{ color: 'var(--text-primary)' }}>
            Running evaluation pipeline…
          </p>
          <p className="text-[12px] mt-1" style={{ color: 'var(--text-secondary)' }}>
            Processing ground truth dataset · Computing IR metrics + LLM judgements
          </p>
        </div>
      )}

      {/* No data */}
      {!hasData && !evalMutation.isPending && (
        <div className="card text-center py-20"
             style={{ borderStyle: 'dashed', borderColor: 'rgba(255,255,255,0.08)' }}>
          <div className="w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-4"
               style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
            <BarChart2 className="w-7 h-7" style={{ color: 'var(--text-secondary)' }} />
          </div>
          <p className="font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>
            No evaluation data yet
          </p>
          <p className="text-[12px]" style={{ color: 'var(--text-secondary)' }}>
            Click "Run Evaluation" to benchmark the retrieval system.
          </p>
        </div>
      )}

      {hasData && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="space-y-5"
        >
          {/* Run meta */}
          <div className="flex items-center gap-3 text-[11px] flex-wrap"
               style={{ color: 'var(--text-secondary)' }}>
            {latest?.run_id && (
              <span className="mono px-2 py-1 rounded-lg"
                    style={{ background: 'rgba(79,140,255,0.1)', color: '#4F8CFF',
                             border: '1px solid rgba(79,140,255,0.2)' }}>
                {latest.run_id}
              </span>
            )}
            {latest?.timestamp && (
              <span>{new Date(latest.timestamp).toLocaleString()}</span>
            )}
            <span className={`badge ${latest?.overall_passed ? 'badge-teal' : 'badge-critical'}`}>
              {latest?.overall_passed ? `✓ All ${metrics.length} passed` : `${passed}/${metrics.length} passed`}
            </span>
            {latest?.num_test_cases && (
              <span>{latest.num_test_cases} test cases</span>
            )}
          </div>

          {/* Metric cards grid — click any card to see "Why this score" */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
            {metrics.map((m, i) => (
              <MetricCard
                key={m.name}
                m={m}
                i={i}
                expanded={expandedCard === m.name}
                onToggle={() => setExpandedCard(prev => prev === m.name ? null : m.name)}
              />
            ))}
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Bar chart */}
            <div className="card">
              <div className="flex items-center gap-2 mb-4">
                <BarChart2 className="w-4 h-4" style={{ color: '#4F8CFF' }} />
                <p className="font-semibold text-[13px]" style={{ color: 'var(--text-primary)' }}>
                  Scores vs Thresholds
                </p>
              </div>
              <ResponsiveContainer width="100%" height={230}>
                <BarChart data={barData} margin={{ top: 4, right: 8, left: -22, bottom: 46 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false}
                                 stroke="rgba(255,255,255,0.04)" />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 10, fill: 'var(--text-secondary)' }}
                    angle={-35}
                    textAnchor="end"
                    height={56}
                  />
                  <YAxis
                    domain={[0, 100]}
                    tick={{ fontSize: 10, fill: '#4B5563' }}
                    tickFormatter={v => `${v}%`}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="score" radius={[6, 6, 0, 0]} maxBarSize={38}>
                    {barData.map((entry, i) => (
                      <Cell
                        key={i}
                        fill={entry.passed ? '#23C6A8' : '#F05A5A'}
                        fillOpacity={0.8}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Radar */}
            <div className="card">
              <div className="flex items-center gap-2 mb-1">
                <Zap className="w-4 h-4" style={{ color: '#9B7FEA' }} />
                <p className="font-semibold text-[13px]" style={{ color: 'var(--text-primary)' }}>
                  Radar Overview
                </p>
              </div>
              <p className="text-[10px] mb-2" style={{ color: '#374151' }}>
                Blue = score · Amber dashed = threshold
              </p>
              <MetricChart metrics={metrics} />
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}
