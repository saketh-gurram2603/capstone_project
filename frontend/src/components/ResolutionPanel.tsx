import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, ChevronUp, Sparkles } from 'lucide-react'
import type { ResolutionOption } from '../api/searchApi'

const RANK_COLORS = [
  { dot: '#4F8CFF', bg: 'rgba(79,140,255,0.08)',  border: 'rgba(79,140,255,0.2)'  },
  { dot: '#9B7FEA', bg: 'rgba(155,127,234,0.08)', border: 'rgba(155,127,234,0.2)' },
  { dot: '#23C6A8', bg: 'rgba(35,198,168,0.08)',  border: 'rgba(35,198,168,0.2)'  },
  { dot: '#9CA7B3', bg: 'rgba(156,167,179,0.06)', border: 'rgba(156,167,179,0.12)'},
]

export default function ResolutionPanel({ options }: { options: ResolutionOption[] }) {
  const [expanded, setExpanded] = useState<number | null>(0)
  if (!options.length) return null

  return (
    <div className="space-y-2">
      {options.map((opt, i) => {
        const isOpen = expanded === i
        const c = RANK_COLORS[Math.min(i, RANK_COLORS.length - 1)]

        return (
          <div key={i}
               className="rounded-xl overflow-hidden transition-all duration-150"
               style={{
                 background: isOpen ? c.bg : 'rgba(30,37,48,0.6)',
                 border: `1px solid ${isOpen ? c.border : 'var(--border)'}`,
               }}>
            <button
              onClick={() => setExpanded(isOpen ? null : i)}
              className="w-full flex items-center justify-between px-4 py-2.5 text-left"
            >
              <div className="flex items-center gap-2.5">
                <span className="w-2 h-2 rounded-full flex-shrink-0"
                      style={{ background: c.dot }} />
                <span className="text-[12px] font-semibold" style={{ color: 'var(--text-primary)' }}>
                  Fix {i + 1}
                </span>
                <span className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>
                  Used <strong style={{ color: 'var(--text-primary)' }}>{opt.occurrence_count}×</strong> historically
                </span>
              </div>
              {isOpen
                ? <ChevronUp   className="w-3.5 h-3.5 flex-shrink-0" style={{ color: 'var(--text-secondary)' }} />
                : <ChevronDown className="w-3.5 h-3.5 flex-shrink-0" style={{ color: 'var(--text-secondary)' }} />
              }
            </button>

            <AnimatePresence>
              {isOpen && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="overflow-hidden"
                >
                  <div className="px-4 pb-4">
                    <div className="divider mb-3" />
                    <p className="text-[12px] leading-relaxed"
                       style={{ color: 'var(--text-secondary)' }}>
                      {opt.resolution_text}
                    </p>
                    {opt.source_incident_ids?.length > 0 && (
                      <p className="mt-2.5 text-[10px] flex items-center gap-1"
                         style={{ color: '#374151' }}>
                        <Sparkles className="w-3 h-3" />
                        Sources: {opt.source_incident_ids.slice(0, 4).join(', ')}
                        {opt.source_incident_ids.length > 4
                          ? ` +${opt.source_incident_ids.length - 4} more`
                          : ''}
                      </p>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )
      })}
    </div>
  )
}
