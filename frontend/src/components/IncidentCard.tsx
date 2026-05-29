import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, ChevronUp, Tag, Wrench } from 'lucide-react'
import type { SearchResultItem } from '../api/searchApi'

interface Props {
  item: SearchResultItem
  rank: number
}

const P_BORDER: Record<string, string> = {
  P1: 'border-p1', P2: 'border-p2', P3: 'border-p3', P4: 'border-p4',
}
const P_BADGE: Record<string, string> = {
  P1: 'badge-critical', P2: 'badge-high', P3: 'badge-medium', P4: 'badge-low',
}
const SCORE_COLOR = (pct: number) =>
  pct >= 75 ? '#23C6A8' : pct >= 50 ? '#4F8CFF' : pct >= 30 ? '#F4B740' : '#4B5563'

export default function IncidentCard({ item, rank }: Props) {
  const [expanded, setExpanded] = useState(false)
  const pBorder    = P_BORDER[item.priority ?? ''] ?? 'border-p4'
  const pBadge     = P_BADGE[item.priority ?? ''] ?? 'badge-low'
  const rawScore   = item.similarity_score ?? 0
  const scorePct   = Math.round(rawScore * 100)
  const scoreLabel = scorePct === 0 && rawScore > 0 ? '<1' : `${scorePct}`
  const scoreCol   = SCORE_COLOR(scorePct)

  return (
    <div
      className={`card-sm hover:shadow-glow transition-all duration-200 cursor-default pl-4 ${pBorder}`}
      style={{ borderLeftWidth: 3 }}
      onMouseEnter={e => {
        ;(e.currentTarget as HTMLElement).style.borderColor = 'rgba(79,140,255,0.25)'
        ;(e.currentTarget as HTMLElement).style.background = 'rgba(36,45,59,0.95)'
      }}
      onMouseLeave={e => {
        ;(e.currentTarget as HTMLElement).style.borderColor = 'var(--border)'
        ;(e.currentTarget as HTMLElement).style.background = 'var(--card)'
      }}
    >
      <div className="flex items-start gap-3">
        {/* Rank */}
        <div className="flex-shrink-0 w-6 h-6 rounded-full text-[11px] font-bold
                        flex items-center justify-center mt-0.5"
             style={{ background: 'var(--surface)', color: 'var(--text-secondary)',
                      border: '1px solid var(--border)' }}>
          {rank}
        </div>

        <div className="flex-1 min-w-0">
          {/* Header row */}
          <div className="flex flex-wrap items-center gap-2 mb-2">
            <span className="mono text-[11px]" style={{ color: 'var(--text-secondary)' }}>
              {item.incident_id}
            </span>
            <h3 className="text-[13px] font-semibold flex-1 truncate"
                style={{ color: 'var(--text-primary)' }}>
              {item.title || item.description?.slice(0, 60)}
            </h3>
            <div className="flex items-center gap-1.5 flex-shrink-0">
              {item.priority && (
                <span className={`badge ${pBadge}`}>{item.priority}</span>
              )}
              {item.category && (
                <span className="badge badge-blue">
                  <Tag className="w-2.5 h-2.5" />
                  {item.category}
                </span>
              )}
            </div>
          </div>

          {/* Score bar */}
          <div className="flex items-center gap-2 mb-2">
            <div className="w-24 score-track">
              <motion.div
                className="score-fill"
                initial={{ width: 0 }}
                animate={{ width: `${scorePct}%` }}
                transition={{ duration: 0.7 }}
                style={{ background: scoreCol }}
              />
            </div>
            <span className="text-[11px] tabular-nums font-semibold"
                  style={{ color: scoreCol }}>
              {scoreLabel}%
            </span>
            <span className="text-[10px]" style={{ color: 'var(--text-secondary)' }}>
              similarity
            </span>
          </div>

          {/* Description */}
          <p className="text-[12px] leading-relaxed line-clamp-2"
             style={{ color: 'var(--text-secondary)' }}>
            {item.description || 'No description available.'}
          </p>

          {/* Expand resolution */}
          {item.resolution_notes && (
            <>
              <button
                onClick={() => setExpanded(!expanded)}
                className="mt-2 btn-ghost text-[11px] p-0"
                style={{ color: '#4F8CFF' }}
              >
                <Wrench className="w-3 h-3" />
                {expanded
                  ? <><ChevronUp className="w-3 h-3" /> Hide resolution</>
                  : <><ChevronDown className="w-3 h-3" /> View resolution</>}
              </button>
              <AnimatePresence>
                {expanded && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden mt-2"
                  >
                    <div className="p-3 rounded-xl text-[12px] leading-relaxed"
                         style={{ background: 'rgba(35,198,168,0.06)',
                                  border: '1px solid rgba(35,198,168,0.15)',
                                  color: 'var(--text-secondary)' }}>
                      <span className="section-label block mb-1.5">Resolution</span>
                      {item.resolution_notes}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
