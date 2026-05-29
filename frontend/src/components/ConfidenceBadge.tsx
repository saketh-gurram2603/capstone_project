import { motion } from 'framer-motion'

interface Props {
  level: 'L1' | 'L2' | 'L3'
  confidence: number
}

const CFG: Record<string, { color: string; bg: string; border: string; label: string }> = {
  L1: { color: '#23C6A8', bg: 'rgba(35,198,168,0.1)',  border: 'rgba(35,198,168,0.25)',  label: 'Auto-Resolved'  },
  L2: { color: '#F4B740', bg: 'rgba(244,183,64,0.1)',  border: 'rgba(244,183,64,0.25)',  label: 'Web-Augmented'  },
  L3: { color: '#F05A5A', bg: 'rgba(240,90,90,0.1)',   border: 'rgba(240,90,90,0.25)',   label: 'Escalated'      },
}

export default function ConfidenceBadge({ level, confidence }: Props) {
  const cfg = CFG[level] ?? CFG.L3
  const pct = Math.round(confidence * 100)

  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg font-semibold text-[12px]"
           style={{ background: cfg.bg, border: `1px solid ${cfg.border}`, color: cfg.color }}>
        <span className="w-1.5 h-1.5 rounded-full" style={{ background: cfg.color }} />
        {level} · {cfg.label}
      </div>

      <div className="flex items-center gap-2">
        <div className="w-28 score-track">
          <motion.div
            className="score-fill"
            initial={{ width: 0 }}
            animate={{ width: `${pct}%` }}
            transition={{ duration: 0.8 }}
            style={{ background: cfg.color }}
          />
        </div>
        <span className="text-[12px] font-bold tabular-nums" style={{ color: cfg.color }}>
          {pct}%
        </span>
      </div>
    </div>
  )
}
