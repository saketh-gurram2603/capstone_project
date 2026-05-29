import { memo } from 'react'
import { Handle, Position } from 'reactflow'
import { motion } from 'framer-motion'
import { CheckCircle, Loader2, Clock, SkipForward, AlertCircle, Zap } from 'lucide-react'

export type AgentStatus = 'idle' | 'running' | 'resolved' | 'escalated' | 'skipped' | 'failed'

export interface AgentNodeData {
  label:       string
  sublabel:    string
  model:       string
  status:      AgentStatus
  confidence?: number
  latency_ms?: number
  isSource?:   boolean
}

const STATUS_CONFIG: Record<AgentStatus, {
  border: string; bg: string; dot: string; icon: React.ReactNode; label: string
}> = {
  idle: {
    border: 'rgba(255,255,255,0.08)',
    bg:     'rgba(30,37,48,0.9)',
    dot:    '#4B5563',
    icon:   <Clock className="w-3.5 h-3.5" style={{ color: '#9CA7B3' }} />,
    label:  'Idle',
  },
  running: {
    border: 'rgba(79,140,255,0.45)',
    bg:     'rgba(79,140,255,0.06)',
    dot:    '#4F8CFF',
    icon:   <Loader2 className="w-3.5 h-3.5 animate-spin" style={{ color: '#4F8CFF' }} />,
    label:  'Processing',
  },
  resolved: {
    border: 'rgba(35,198,168,0.45)',
    bg:     'rgba(35,198,168,0.06)',
    dot:    '#23C6A8',
    icon:   <CheckCircle className="w-3.5 h-3.5" style={{ color: '#23C6A8' }} />,
    label:  'Resolved',
  },
  escalated: {
    border: 'rgba(244,183,64,0.45)',
    bg:     'rgba(244,183,64,0.06)',
    dot:    '#F4B740',
    icon:   <Zap className="w-3.5 h-3.5" style={{ color: '#F4B740' }} />,
    label:  'Escalated',
  },
  skipped: {
    border: 'rgba(255,255,255,0.04)',
    bg:     'rgba(13,17,23,0.6)',
    dot:    '#374151',
    icon:   <SkipForward className="w-3.5 h-3.5" style={{ color: '#374151' }} />,
    label:  'Skipped',
  },
  failed: {
    border: 'rgba(240,90,90,0.45)',
    bg:     'rgba(240,90,90,0.06)',
    dot:    '#F05A5A',
    icon:   <AlertCircle className="w-3.5 h-3.5" style={{ color: '#F05A5A' }} />,
    label:  'Failed',
  },
}

function AgentNode({ data }: { data: AgentNodeData }) {
  const cfg   = STATUS_CONFIG[data.status]
  const isActive = data.status === 'running' || data.status === 'resolved' || data.status === 'escalated'
  const opacity = data.status === 'skipped' ? 0.45 : 1

  return (
    <motion.div
      animate={{ opacity, scale: data.status === 'running' ? [1, 1.015, 1] : 1 }}
      transition={{ scale: { repeat: Infinity, duration: 1.5 } }}
      style={{
        width: 180,
        background: cfg.bg,
        border: `1.5px solid ${cfg.border}`,
        borderRadius: 14,
        padding: '14px 16px',
        boxShadow: isActive ? `0 0 20px ${cfg.border}` : 'none',
        cursor: 'default',
        position: 'relative',
      }}
    >
      {!data.isSource && (
        <Handle
          type="target"
          position={Position.Left}
          style={{ background: cfg.dot, width: 8, height: 8, border: '2px solid var(--bg-primary)' }}
        />
      )}

      {/* Status indicator + label */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5">
          {data.status === 'running' && (
            <span className="pulse-dot w-2 h-2 rounded-full inline-block" style={{ background: cfg.dot }} />
          )}
          {data.status !== 'running' && (
            <span className="w-2 h-2 rounded-full inline-block" style={{ background: cfg.dot }} />
          )}
          <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: cfg.dot }}>
            {cfg.label}
          </span>
        </div>
        {cfg.icon}
      </div>

      {/* Name */}
      <h4 className="text-[13px] font-bold mb-0.5" style={{ color: 'var(--text-primary)' }}>
        {data.label}
      </h4>
      <p className="text-[11px] mb-3" style={{ color: 'var(--text-secondary)' }}>
        {data.sublabel}
      </p>

      {/* Divider */}
      <div className="divider mb-2.5" />

      {/* Model badge */}
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <span className="mono text-[10px] px-2 py-0.5 rounded-md"
              style={{ background: 'rgba(255,255,255,0.06)', color: 'var(--text-secondary)' }}>
          {data.model}
        </span>
        {data.confidence !== undefined && (
          <span className="text-[11px] font-semibold tabular-nums" style={{ color: '#23C6A8' }}>
            {Math.round(data.confidence * 100)}%
          </span>
        )}
      </div>

      {data.latency_ms !== undefined && (
        <p className="text-[10px] mt-1.5 tabular-nums" style={{ color: 'var(--text-secondary)' }}>
          ⚡ {data.latency_ms.toFixed(0)} ms
        </p>
      )}

      <Handle
        type="source"
        position={Position.Right}
        style={{ background: cfg.dot, width: 8, height: 8, border: '2px solid var(--bg-primary)' }}
      />
    </motion.div>
  )
}

export default memo(AgentNode)
