import { useState, useCallback, useEffect } from 'react'
import { useLocation } from 'react-router-dom'
import { useMutation, useQuery } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import ReactFlow, {
  Node, Edge, Background, Controls,
  useNodesState, useEdgesState, MarkerType,
} from 'reactflow'
import 'reactflow/dist/style.css'
import {
  GitBranch, AlertTriangle, Ticket, ChevronDown,
  ChevronRight, X, Loader2, Clipboard,
} from 'lucide-react'
import { runTriage, getEscalations, type TriageResult } from '../api/triageApi'
import ConfidenceBadge from '../components/ConfidenceBadge'
import AgentNode, { type AgentStatus } from '../components/AgentNode'

// ── React Flow node types ─────────────────────────────────
const nodeTypes = { agentNode: AgentNode }

// ── Initial graph state ───────────────────────────────────
function makeNodes(
  l1Status: AgentStatus = 'idle',
  l2Status: AgentStatus = 'idle',
  l3Status: AgentStatus = 'idle',
  result?: TriageResult,
): Node[] {
  return [
    {
      id: 'input',
      type: 'agentNode',
      position: { x: 20, y: 100 },
      data: {
        label:    'Input Query',
        sublabel: 'Incident description',
        model:    'user input',
        status:   (l1Status !== 'idle' ? 'resolved' : 'idle') as AgentStatus,
        isSource: true,
      },
    },
    {
      id: 'l1',
      type: 'agentNode',
      position: { x: 240, y: 100 },
      data: {
        label:      'L1 Triage Agent',
        sublabel:   'KB hybrid search',
        model:      'gpt-4o-mini',
        status:     l1Status,
        confidence: result?.escalation_level === 'L1' ? result.confidence : undefined,
        latency_ms: result?.escalation_level === 'L1' ? result.latency_ms * 0.5 : undefined,
      },
    },
    {
      id: 'l2',
      type: 'agentNode',
      position: { x: 460, y: 100 },
      data: {
        label:      'L2 Analysis Agent',
        sublabel:   'Web-augmented search',
        model:      'gpt-4o',
        status:     l2Status,
        confidence: result?.escalation_level === 'L2' ? result.confidence : undefined,
        latency_ms: result?.escalation_level === 'L2' ? result.latency_ms * 0.6 : undefined,
      },
    },
    {
      id: 'l3',
      type: 'agentNode',
      position: { x: 680, y: 100 },
      data: {
        label:      'L3 Specialist',
        sublabel:   'Escalation queue',
        model:      'rule-based',
        status:     l3Status,
        confidence: result?.escalation_level === 'L3' ? result.confidence : undefined,
        latency_ms: result?.escalation_level === 'L3' ? result.latency_ms * 0.2 : undefined,
      },
    },
  ]
}

const EDGE_BASE: Omit<Edge, 'id' | 'source' | 'target'> = {
  type: 'smoothstep',
  animated: false,
  markerEnd: { type: MarkerType.ArrowClosed, color: 'rgba(255,255,255,0.15)', width: 16, height: 16 },
  style: { stroke: 'rgba(255,255,255,0.12)', strokeWidth: 1.5 },
}

const initialEdges: Edge[] = [
  { id: 'e-in-l1', source: 'input', target: 'l1', ...EDGE_BASE },
  { id: 'e-l1-l2', source: 'l1', target: 'l2', ...EDGE_BASE },
  { id: 'e-l2-l3', source: 'l2', target: 'l3', ...EDGE_BASE },
]

// ── Triage page ───────────────────────────────────────────
export default function TriagePage() {
  const [description, setDescription] = useState('')
  const [impact,  setImpact]  = useState('')
  const [urgency, setUrgency] = useState('')
  const [showEscalations, setShowEscalations] = useState(false)
  const [copied, setCopied] = useState(false)

  const location = useLocation()

  // Pre-fill description when navigated from Search page ("Triage Top Incident")
  useEffect(() => {
    const incoming = (location.state as { description?: string } | null)?.description
    if (incoming) {
      setDescription(incoming)
    }
  }, [])

  const [nodes, setNodes, onNodesChange] = useNodesState(makeNodes())
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)

  const triage = useMutation({
    mutationFn: () => runTriage({ description, impact: (impact || undefined) as any, urgency: (urgency || undefined) as any }),
  })

  const escalations = useQuery({
    queryKey: ['escalations'],
    queryFn: () => getEscalations(),
    enabled: showEscalations,
  })

  const result: TriageResult | undefined = triage.data

  // Animate nodes based on result
  useEffect(() => {
    if (!triage.isPending && !result) {
      setNodes(makeNodes())
      return
    }
    if (triage.isPending) {
      setNodes(makeNodes('running'))
      return
    }
    if (!result) return

    const level = result.escalation_level

    // Animate through the pipeline with delays
    const animateL1 = () => setNodes(makeNodes('running', 'idle', 'idle'))
    const animateL1done = () => {
      if (level === 'L1') setNodes(makeNodes('resolved', 'skipped', 'skipped', result))
      else setNodes(makeNodes('escalated', 'running', 'idle'))
    }
    const animateL2done = () => {
      if (level === 'L2') setNodes(makeNodes('escalated', 'resolved', 'skipped', result))
      else setNodes(makeNodes('escalated', 'escalated', 'running'))
    }
    const animateL3done = () => setNodes(makeNodes('escalated', 'escalated', 'resolved', result))

    animateL1()
    const t1 = setTimeout(animateL1done, 600)
    const t2 = level !== 'L1' ? setTimeout(animateL2done, 1400) : 0
    const t3 = level === 'L3' ? setTimeout(animateL3done, 2200) : 0

    return () => { clearTimeout(t1); if (t2) clearTimeout(t2); if (t3) clearTimeout(t3) }
  }, [triage.isPending, result])

  // Update edge colors based on active path
  useEffect(() => {
    if (!result) return
    const activeColor = (active: boolean) => active ? '#4F8CFF' : 'rgba(255,255,255,0.1)'
    const l1Active = true
    const l2Active = result.escalation_level === 'L2' || result.escalation_level === 'L3'
    const l3Active = result.escalation_level === 'L3'
    setEdges([
      { id: 'e-in-l1', source: 'input', target: 'l1', ...EDGE_BASE,
        animated: l1Active, style: { ...EDGE_BASE.style, stroke: activeColor(l1Active) },
        markerEnd: { ...EDGE_BASE.markerEnd as any, color: activeColor(l1Active) } },
      { id: 'e-l1-l2', source: 'l1', target: 'l2', ...EDGE_BASE,
        animated: l2Active, style: { ...EDGE_BASE.style, stroke: activeColor(l2Active) },
        markerEnd: { ...EDGE_BASE.markerEnd as any, color: activeColor(l2Active) } },
      { id: 'e-l2-l3', source: 'l2', target: 'l3', ...EDGE_BASE,
        animated: l3Active, style: { ...EDGE_BASE.style, stroke: activeColor(l3Active) },
        markerEnd: { ...EDGE_BASE.markerEnd as any, color: activeColor(l3Active) } },
    ])
  }, [result])

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (description.trim().length < 10) return
    setNodes(makeNodes())
    setEdges(initialEdges)
    triage.reset()
    triage.mutate()
  }

  function copyTicket() {
    if (result?.escalation_ticket_id) {
      navigator.clipboard.writeText(result.escalation_ticket_id)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  return (
    <div className="flex gap-5 h-full" style={{ minHeight: 'calc(100vh - 56px - 48px)' }}>
      {/* ── Left: Form + Result ───────────────────────────── */}
      <div className="flex flex-col gap-4" style={{ width: 340, flexShrink: 0 }}>
        {/* Form */}
        <form onSubmit={handleSubmit} className="card space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <GitBranch className="w-4 h-4" style={{ color: 'var(--accent-blue)' }} />
              <span className="font-semibold text-[13px]" style={{ color: 'var(--text-primary)' }}>
                Triage Incident
              </span>
            </div>
            <button
              type="button"
              className="btn-ghost text-[12px]"
              onClick={() => setShowEscalations(!showEscalations)}
              style={{ color: showEscalations ? '#F05A5A' : 'var(--text-secondary)' }}
            >
              <Ticket className="w-3.5 h-3.5" />
              Tickets
            </button>
          </div>

          <div>
            <label className="section-label block mb-1.5">
              Description <span style={{ color: '#F05A5A' }}>*</span>
            </label>
            <textarea
              className="input"
              style={{ resize: 'none', minHeight: 100, lineHeight: 1.6 }}
              placeholder="Describe the incident in detail..."
              value={description}
              onChange={e => setDescription(e.target.value)}
              rows={4}
            />
            <p className="text-[10px] mt-1"
               style={{ color: description.length < 10 ? 'var(--warning)' : '#23C6A8' }}>
              {description.length < 10
                ? `${10 - description.length} more chars needed`
                : '✓ Ready to triage'}
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3">
            {[
              { label: 'Impact', value: impact, set: setImpact },
              { label: 'Urgency', value: urgency, set: setUrgency },
            ].map(({ label, value, set }) => (
              <div key={label}>
                <label className="section-label block mb-1.5">{label}</label>
                <select className="input" value={value} onChange={e => set(e.target.value)}>
                  <option value="">Not specified</option>
                  <option value="High">High</option>
                  <option value="Medium">Medium</option>
                  <option value="Low">Low</option>
                </select>
              </div>
            ))}
          </div>

          <button
            type="submit"
            className="btn-primary w-full justify-center py-2.5"
            disabled={description.trim().length < 10 || triage.isPending}
          >
            {triage.isPending ? (
              <><Loader2 className="w-4 h-4 animate-spin" /> Analysing…</>
            ) : (
              <><GitBranch className="w-4 h-4" /> Run Triage</>
            )}
          </button>
        </form>

        {/* Error */}
        {triage.isError && (
          <div className="card-sm flex items-center gap-2 text-[12px]"
               style={{ color: '#F05A5A', borderColor: 'rgba(240,90,90,0.2)',
                        background: 'rgba(240,90,90,0.05)' }}>
            <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" />
            {(triage.error as Error).message}
          </div>
        )}

        {/* Result card */}
        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              className="card space-y-4 overflow-y-auto"
              style={{ flex: 1 }}
            >
              <ConfidenceBadge level={result.escalation_level} confidence={result.confidence} />

              {/* Ticket ID */}
              {result.escalation_ticket_id && (
                <div className="flex items-center gap-2 p-2.5 rounded-xl"
                     style={{ background: 'rgba(240,90,90,0.08)', border: '1px solid rgba(240,90,90,0.2)' }}>
                  <Ticket className="w-3.5 h-3.5 flex-shrink-0" style={{ color: '#F05A5A' }} />
                  <span className="mono text-[12px] font-bold flex-1" style={{ color: '#F05A5A' }}>
                    {result.escalation_ticket_id}
                  </span>
                  <button className="btn-ghost p-1" onClick={copyTicket}>
                    <Clipboard className="w-3 h-3" />
                  </button>
                  {copied && <span className="text-[10px]" style={{ color: '#23C6A8' }}>Copied!</span>}
                </div>
              )}

              <div className="divider" />

              {/* Final answer */}
              <div>
                <p className="section-label mb-2">Resolution</p>
                <div className="text-[12px] leading-relaxed whitespace-pre-wrap"
                     style={{ color: 'var(--text-secondary)' }}>
                  {result.final_answer}
                </div>
              </div>

              {/* L1 output — always visible when escalated so you can see what L1 thought */}
              {result.l1_summary && result.escalation_level !== 'L1' && (
                <div>
                  <p className="section-label mb-1.5 flex items-center gap-1.5">
                    <span className="w-4 h-4 rounded flex items-center justify-center text-[9px] font-bold flex-shrink-0"
                          style={{ background: 'rgba(35,198,168,0.2)', color: '#23C6A8' }}>
                      L1
                    </span>
                    What L1 found in the knowledge base
                    <span className="ml-auto font-normal" style={{ color: 'rgba(255,255,255,0.25)' }}>
                      below 80% threshold — escalated to L2
                    </span>
                  </p>
                  <div className="p-3 rounded-xl text-[11px] leading-relaxed"
                       style={{ background: 'rgba(35,198,168,0.05)',
                                border: '1px solid rgba(35,198,168,0.18)',
                                color: 'var(--text-secondary)' }}>
                    {result.l1_summary}
                  </div>
                </div>
              )}

              {/* L2 synthesis — always visible when escalated to L3 */}
              {result.l2_synthesis && result.escalation_level === 'L3' && (
                <div>
                  <p className="section-label mb-1.5 flex items-center gap-1.5">
                    <span className="w-4 h-4 rounded flex items-center justify-center text-[9px] font-bold flex-shrink-0"
                          style={{ background: 'rgba(244,183,64,0.2)', color: '#F4B740' }}>
                      L2
                    </span>
                    What L2 found with web search
                    <span className="ml-auto font-normal" style={{ color: 'rgba(255,255,255,0.25)' }}>
                      below 55% threshold — escalated to L3
                    </span>
                  </p>
                  <div className="p-3 rounded-xl text-[11px] leading-relaxed"
                       style={{ background: 'rgba(244,183,64,0.05)',
                                border: '1px solid rgba(244,183,64,0.18)',
                                color: 'var(--text-secondary)' }}>
                    {result.l2_synthesis}
                  </div>
                </div>
              )}

              {/* Meta */}
              <div className="text-[11px] space-y-1" style={{ color: 'var(--text-secondary)' }}>
                {result.priority && (
                  <p>Priority: <strong style={{ color: 'var(--text-primary)' }}>{result.priority}</strong></p>
                )}
                <p>Model: <span className="mono">{result.model_used}</span></p>
                <p>Latency: <span className="tabular-nums">{result.latency_ms.toFixed(0)} ms</span></p>
                {result.fallback_used && (
                  <span className="badge badge-high">⚠ Fallback used</span>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Escalation tickets panel */}
        <AnimatePresence>
          {showEscalations && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="card overflow-hidden"
            >
              <div className="flex items-center justify-between mb-3">
                <p className="font-semibold text-[13px] flex items-center gap-1.5"
                   style={{ color: 'var(--text-primary)' }}>
                  <Ticket className="w-4 h-4" style={{ color: '#F05A5A' }} />
                  Escalation Tickets
                </p>
                <button className="btn-ghost p-1" onClick={() => setShowEscalations(false)}>
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
              <div className="divider mb-3" />
              {escalations.isPending && (
                <p className="text-[12px]" style={{ color: 'var(--text-secondary)' }}>Loading…</p>
              )}
              {escalations.data?.tickets.length === 0 && (
                <p className="text-[12px] text-center py-4" style={{ color: 'var(--text-secondary)' }}>
                  No escalation tickets yet.
                </p>
              )}
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {escalations.data?.tickets.map(t => (
                  <div key={t.ticket_id} className="card-sm space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="mono font-bold" style={{ color: '#4F8CFF' }}>{t.ticket_id}</span>
                      <span className="badge badge-critical">{t.status}</span>
                    </div>
                    <p className="text-[11px] line-clamp-1" style={{ color: 'var(--text-secondary)' }}>
                      {t.description}
                    </p>
                    <p className="text-[10px]" style={{ color: '#374151' }}>
                      {new Date(t.created_at).toLocaleString()}
                    </p>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ── Right: React Flow canvas ──────────────────────── */}
      <div className="flex-1 rounded-panel overflow-hidden relative"
           style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)',
                    borderRadius: 20, minHeight: 420 }}>
        {/* Canvas label */}
        <div className="absolute top-4 left-4 z-10 flex items-center gap-2">
          <span className="section-label">Agent Pipeline</span>
          {triage.isPending && (
            <span className="flex items-center gap-1 text-[11px]" style={{ color: '#4F8CFF' }}>
              <Loader2 className="w-3 h-3 animate-spin" /> Processing
            </span>
          )}
          {result && (
            <span className="badge" style={{
              background: result.escalation_level === 'L1' ? 'rgba(35,198,168,0.15)' :
                          result.escalation_level === 'L2' ? 'rgba(244,183,64,0.15)' :
                          'rgba(240,90,90,0.15)',
              color: result.escalation_level === 'L1' ? '#23C6A8' :
                     result.escalation_level === 'L2' ? '#F4B740' : '#F05A5A',
            }}>
              {result.escalation_level} · {Math.round(result.confidence * 100)}% confidence
            </span>
          )}
        </div>

        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.3 }}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          panOnScroll={false}
          zoomOnScroll={false}
        >
          <Background color="rgba(255,255,255,0.03)" gap={28} />
          <Controls showInteractive={false} />
        </ReactFlow>

        {/* Legend */}
        <div className="absolute bottom-4 right-4 flex items-center gap-3"
             style={{ color: 'var(--text-secondary)' }}>
          {[
            { color: '#374151', label: 'Idle' },
            { color: '#4F8CFF', label: 'Running' },
            { color: '#23C6A8', label: 'Resolved' },
            { color: '#F4B740', label: 'Escalated' },
          ].map(({ color, label }) => (
            <div key={label} className="flex items-center gap-1 text-[10px]">
              <span className="w-2 h-2 rounded-full" style={{ background: color }} />
              {label}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
