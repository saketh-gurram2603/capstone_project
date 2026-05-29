import {
  RadarChart, PolarGrid, PolarAngleAxis, Radar,
  ResponsiveContainer, Tooltip,
} from 'recharts'
import type { MetricScore } from '../api/evaluationApi'

const LABEL_MAP: Record<string, string> = {
  ndcg_at_k:                'NDCG@10',
  map_at_k:                 'MAP@10',
  recall_at_k:              'Recall@10',
  precision_at_k:           'P@10',
  avg_faithfulness:         'Faithfulness',
  avg_answer_relevancy:     'Relevancy',
  avg_contextual_precision: 'Ctx Prec.',
}

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="card-sm text-[11px]">
      {payload.map((p: any) => (
        <p key={p.name} className="font-medium" style={{ color: p.color }}>
          {p.name}: {p.value}%
        </p>
      ))}
    </div>
  )
}

export default function MetricChart({ metrics }: { metrics: MetricScore[] }) {
  if (!metrics.length) return null
  const data = metrics.map(m => ({
    metric:    LABEL_MAP[m.name] ?? m.name,
    score:     Math.round(m.score * 100),
    threshold: Math.round(m.threshold * 100),
  }))
  return (
    <ResponsiveContainer width="100%" height={260}>
      <RadarChart data={data} outerRadius="72%">
        <PolarGrid stroke="rgba(255,255,255,0.05)" />
        <PolarAngleAxis dataKey="metric"
                        tick={{ fontSize: 10, fill: 'var(--text-secondary)' }} />
        <Radar name="Score"     dataKey="score"
               stroke="#4F8CFF" fill="#4F8CFF" fillOpacity={0.18} strokeWidth={2} />
        <Radar name="Threshold" dataKey="threshold"
               stroke="#F4B740" fill="#F4B740" fillOpacity={0.05}
               strokeDasharray="4 2" strokeWidth={1.5} />
        <Tooltip content={<CustomTooltip />} />
      </RadarChart>
    </ResponsiveContainer>
  )
}
