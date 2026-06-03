import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  BarChart2, CheckCircle, XCircle, Play, RefreshCw,
  TrendingUp, Target, Brain, Zap, X, Lightbulb, Clock,
} from 'lucide-react'
import { getLatestMetrics, runEvaluation, type MetricScore } from '../api/evaluationApi'
import MetricChart from '../components/MetricChart'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts'

// ── Metric metadata ───────────────────────────────────────────────────────────

const LABEL_MAP: Record<string, string> = {
  ndcg_at_k:                   'NDCG@10',
  map_at_k:                    'MAP@10',
  recall_at_k:                 'Recall@10',
  precision_at_k:              'P@10',
  avg_faithfulness:            'Faithfulness',
  avg_answer_relevancy:        'Relevancy',
  avg_contextual_precision:    'Ctx Precision',
  fix_accuracy:                'Fix Accuracy',
  resolution_time_mae_hours:   'Res. Time MAE',
}

const ICONS: Record<string, typeof TrendingUp> = {
  ndcg_at_k:                 TrendingUp,
  map_at_k:                  Target,
  recall_at_k:               BarChart2,
  precision_at_k:            BarChart2,
  avg_faithfulness:          Brain,
  avg_answer_relevancy:      Brain,
  avg_contextual_precision:  Brain,
  fix_accuracy:              CheckCircle,
  resolution_time_mae_hours: Clock,
}

const DESC: Record<string, string> = {
  ndcg_at_k:                 'Ranking quality',
  map_at_k:                  'Avg precision',
  recall_at_k:               'Coverage rate',
  precision_at_k:            'Result accuracy',
  avg_faithfulness:          'LLM-as-Judge',
  avg_answer_relevancy:      'LLM-as-Judge',
  avg_contextual_precision:  'LLM-as-Judge',
  fix_accuracy:              'Custom metric',
  resolution_time_mae_hours: 'Custom metric',
}

// Metrics where the score is in hours (not a 0–1 fraction)
const HOUR_METRICS = new Set(['resolution_time_mae_hours'])

// Improvement tips shown in the popup per metric
const TIPS: Record<string, string[]> = {
  ndcg_at_k: [
    'Ensure ground truth relevant IDs cover all semantically similar incidents in the expanded dataset.',
    'Tune RRF fusion weights to better balance BM25 and vector search contributions.',
    'Lower the L1 confidence threshold slightly to surface more relevant candidates.',
  ],
  map_at_k: [
    'Update the ground truth dataset to mark newly added incidents as relevant for each query.',
    'Increase the number of ground truth relevant IDs per test case from 4 to 6–8.',
    'Improve BM25 tokenisation to better handle technical abbreviations.',
  ],
  recall_at_k: [
    'Increase adaptive-K (try k_max = 30) to retrieve more candidates before reranking.',
    'Add more diverse descriptions to the dataset for underrepresented problem types.',
    'Ensure all semantically relevant incidents in the expanded dataset are tagged in the ground truth.',
  ],
  precision_at_k: [
    'Reduce TOP_K_FINAL from 10 to 5 — P@5 is naturally higher when ground truth has ~4 relevant docs.',
    'Improve cross-encoder reranker precision by fine-tuning on domain-specific incident pairs.',
    'Apply a minimum similarity score threshold (≥ 0.45) to filter low-confidence tail results.',
  ],
  avg_faithfulness: [
    'Ensure retrieved context always includes the full resolution_notes, not just description.',
    'Increase the number of context documents passed to the LLM judge from 3 to 5.',
    'Reframe the L1 system prompt to explicitly stay grounded in retrieved incidents.',
  ],
  avg_answer_relevancy: [
    'Add more diverse problem types to the dataset to improve query-answer alignment.',
    'Tune the L1 prompt to answer in terms of actionable resolution steps, not summaries.',
    'Increase retrieval diversity by enabling hybrid mode (BM25 + vector) for all queries.',
  ],
  avg_contextual_precision: [
    'Improve metadata filtering — category, impact, and priority filters reduce irrelevant context.',
    'Tighten the cross-encoder reranker threshold to only pass highly relevant docs to the LLM.',
    'Consider adding incident type tags (network/storage/database) to the Qdrant payload.',
  ],
  fix_accuracy: [
    'Update ground truth expected_resolution_keywords to match the exact language in resolution_notes.',
    'Add more varied keyword aliases per fix type (e.g. "cache" → also "CDN", "edge cache").',
    'After re-ingesting the expanded dataset, re-run evaluation to see the improvement.',
  ],
  resolution_time_mae_hours: [
    'Re-run setup_local_qdrant.py after dataset expansion to populate resolution_hours in Qdrant.',
    'Add more historical incidents with accurate opened_at / resolved_at timestamps.',
    'Group incidents by category and use category-specific resolution time models for better predictions.',
  ],
}

// ── Metric display helpers ────────────────────────────────────────────────────

function formatScore(m: MetricScore): string {
  if (HOUR_METRICS.has(m.name)) return `${m.score.toFixed(1)} hrs`
  return `${Math.round(m.score * 100)}%`
}

function formatThreshold(m: MetricScore): string {
  if (HOUR_METRICS.has(m.name)) return 'display only'
  return `${Math.round(m.threshold * 100)}%`
}

// ── Metric detail popup ───────────────────────────────────────────────────────

function MetricPopup({ m, onClose }: { m: MetricScore; onClose: () => void }) {
  const color  = m.passed ? '#23C6A8' : '#F05A5A'
  const tips   = TIPS[m.name] ?? []

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }}
      onClick={onClose}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 16 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 16 }}
        transition={{ duration: 0.18 }}
        className="card w-full max-w-md relative"
        style={{ maxHeight: '80vh', overflowY: 'auto' }}
        onClick={e => e.stopPropagation()}
      >
        {/* Close */}
        <button
          onClick={onClose}
          className="absolute top-3 right-3 p-1 rounded-lg btn-ghost"
        >
          <X className="w-4 h-4" />
        </button>

        {/* Header */}
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center"
               style={{ background: `${color}1a` }}>
            {m.passed
              ? <CheckCircle className="w-5 h-5" style={{ color }} />
              : <XCircle    className="w-5 h-5" style={{ color }} />}
          </div>
          <div>
            <p className="text-[16px] font-bold" style={{ color: 'var(--text-primary)' }}>
              {LABEL_MAP[m.name] ?? m.name}
            </p>
            <p className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>
              {DESC[m.name] ?? ''}
            </p>
          </div>
        </div>

        {/* Score + threshold row */}
        <div className="flex items-center gap-4 mb-4 p-3 rounded-xl"
             style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border)' }}>
          <div className="text-center flex-1">
            <p className="text-[28px] font-bold tabular-nums" style={{ color }}>
              {formatScore(m)}
            </p>
            <p className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>Current score</p>
          </div>
          <div className="w-px h-10" style={{ background: 'var(--border)' }} />
          <div className="text-center flex-1">
            <p className="text-[22px] font-bold tabular-nums" style={{ color: 'var(--text-secondary)' }}>
              {formatThreshold(m)}
            </p>
            <p className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>Threshold</p>
          </div>
          <div className="w-px h-10" style={{ background: 'var(--border)' }} />
          <div className="text-center flex-1">
            <span className={`badge ${m.passed ? 'badge-teal' : 'badge-critical'}`}>
              {m.passed ? 'PASSED' : 'FAILED'}
            </span>
          </div>
        </div>

        {/* Why this score */}
        {m.reason && (
          <div className="mb-4">
            <p className="text-[12px] font-semibold mb-1.5 flex items-center gap-1.5"
               style={{ color: 'var(--text-primary)' }}>
              <BarChart2 className="w-3.5 h-3.5" style={{ color: '#4F8CFF' }} />
              Why this score
            </p>
            <p className="text-[12px] p-3 rounded-lg leading-relaxed"
               style={{ background: 'rgba(255,255,255,0.04)', color: 'var(--text-secondary)',
                        border: '1px solid var(--border)' }}>
              {m.reason}
            </p>
          </div>
        )}

        {/* How to improve */}
        {tips.length > 0 && (
          <div>
            <p className="text-[12px] font-semibold mb-1.5 flex items-center gap-1.5"
               style={{ color: 'var(--text-primary)' }}>
              <Lightbulb className="w-3.5 h-3.5" style={{ color: '#F4B740' }} />
              How to improve
            </p>
            <div className="space-y-2">
              {tips.map((tip, i) => (
                <div key={i} className="flex items-start gap-2 text-[12px]"
                     style={{ color: 'var(--text-secondary)' }}>
                  <span className="mt-0.5 w-4 h-4 rounded-full flex items-center justify-center flex-shrink-0 text-[10px] font-bold"
                        style={{ background: 'rgba(244,183,64,0.15)', color: '#F4B740' }}>
                    {i + 1}
                  </span>
                  {tip}
                </div>
              ))}
            </div>
          </div>
        )}
      </motion.div>
    </motion.div>
  )
}

// ── Metric card (click → opens popup) ────────────────────────────────────────

function MetricCard({ m, i, onClick }: {
  m: MetricScore; i: number; onClick: () => void
}) {
  const color = m.passed ? '#23C6A8' : '#F05A5A'
  const Icon  = ICONS[m.name] ?? TrendingUp

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: i * 0.06 }}
      className="stat-card cursor-pointer select-none"
      style={{ borderColor: m.passed ? 'rgba(35,198,168,0.2)' : 'rgba(240,90,90,0.2)' }}
      onClick={onClick}
      title="Click for details & improvement tips"
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
        {formatScore(m)}
      </p>
      <p className="text-[11px] font-semibold" style={{ color: 'var(--text-primary)' }}>
        {LABEL_MAP[m.name] ?? m.name}
      </p>
      <p className="text-[10px] mb-3" style={{ color: 'var(--text-secondary)' }}>
        {DESC[m.name] ?? ''}
      </p>

      {!HOUR_METRICS.has(m.name) && (
        <>
          <div className="score-track">
            <motion.div
              className="score-fill"
              initial={{ width: 0 }}
              animate={{ width: `${Math.round(m.score * 100)}%` }}
              transition={{ duration: 0.8, delay: i * 0.06 + 0.1 }}
              style={{ background: color }}
            />
          </div>
          <p className="text-[10px] mt-1.5" style={{ color: '#374151' }}>
            Threshold: {formatThreshold(m)}
          </p>
        </>
      )}

      {HOUR_METRICS.has(m.name) && (
        <p className="text-[10px] mt-1" style={{ color: '#374151' }}>
          Lower is better · display only
        </p>
      )}

      {/* Click hint */}
      <p className="text-[9px] mt-2" style={{ color: 'rgba(255,255,255,0.2)' }}>
        Click for details
      </p>
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

// ── Page ─────────────────────────────────────────────────────────────────────

export default function AnalyticsPage() {
  const [runLlmJudge, setRunLlmJudge] = useState(false)
  const [activeMetric, setActiveMetric] = useState<MetricScore | null>(null)

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

  // Separate hour-based metrics for the bar chart (can't plot % alongside hrs)
  const barData = metrics
    .filter(m => !HOUR_METRICS.has(m.name))
    .map(m => ({
      name:      LABEL_MAP[m.name] ?? m.name,
      score:     Math.round(m.score * 100),
      threshold: Math.round(m.threshold * 100),
      passed:    m.passed,
    }))

  return (
    <>
      {/* ── Detail popup ──────────────────────────────── */}
      <AnimatePresence>
        {activeMetric && (
          <MetricPopup m={activeMetric} onClose={() => setActiveMetric(null)} />
        )}
      </AnimatePresence>

      <div className="space-y-5">
        {/* ── Controls ─────────────────────────────────── */}
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div className="flex items-center gap-3">
            <label className="flex items-center gap-2 cursor-pointer btn-surface"
                   style={{ userSelect: 'none' }}>
              <input type="checkbox" checked={runLlmJudge}
                     onChange={e => setRunLlmJudge(e.target.checked)}
                     className="rounded" style={{ accentColor: 'var(--accent-blue)' }} />
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
            <button className="btn-primary" onClick={() => evalMutation.mutate()}
                    disabled={evalMutation.isPending}>
              {evalMutation.isPending
                ? <><RefreshCw className="w-4 h-4 animate-spin" /> Running…</>
                : <><Play className="w-4 h-4" /> Run Evaluation</>}
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

        {/* Running */}
        {evalMutation.isPending && (
          <div className="card text-center py-16">
            <RefreshCw className="w-7 h-7 animate-spin mx-auto mb-3" style={{ color: '#4F8CFF' }} />
            <p className="font-medium" style={{ color: 'var(--text-primary)' }}>
              Running evaluation pipeline…
            </p>
            <p className="text-[12px] mt-1" style={{ color: 'var(--text-secondary)' }}>
              Processing ground truth · Computing IR metrics + LLM judgements
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
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-5">
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
              {latest?.timestamp && <span>{new Date(latest.timestamp).toLocaleString()}</span>}
              <span className={`badge ${latest?.overall_passed ? 'badge-teal' : 'badge-critical'}`}>
                {latest?.overall_passed
                  ? `✓ All ${metrics.length} passed`
                  : `${passed}/${metrics.length} passed`}
              </span>
              {latest?.num_test_cases && <span>{latest.num_test_cases} test cases</span>}
              <span className="text-[10px] opacity-60">Click any card for details & improvement tips</span>
            </div>

            {/* Metric cards grid — click opens popup, NO inline expansion */}
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
              {metrics.map((m, i) => (
                <MetricCard
                  key={m.name}
                  m={m}
                  i={i}
                  onClick={() => setActiveMetric(m)}
                />
              ))}
            </div>

            {/* Charts — only for non-hour metrics */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
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
                    <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--text-secondary)' }}
                           angle={-35} textAnchor="end" height={56} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: '#4B5563' }}
                           tickFormatter={v => `${v}%`} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="score" radius={[6, 6, 0, 0]} maxBarSize={38}>
                      {barData.map((entry, i) => (
                        <Cell key={i} fill={entry.passed ? '#23C6A8' : '#F05A5A'} fillOpacity={0.8} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

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
                <MetricChart metrics={metrics.filter(m => !HOUR_METRICS.has(m.name))} />
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </>
  )
}
