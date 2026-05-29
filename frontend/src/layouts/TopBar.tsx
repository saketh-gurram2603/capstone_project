import { useLocation } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { Menu, Circle } from 'lucide-react'
import { useUIStore } from '../store/uiStore'

const TITLES: Record<string, { label: string; desc: string }> = {
  '/':          { label: 'Incident Search',  desc: 'Hybrid BM25 + semantic retrieval · cross-encoder reranking' },
  '/triage':    { label: 'Agent Triage',     desc: 'L1 → L2 → L3 confidence-gated routing pipeline'            },
  '/analytics': { label: 'Evaluation Suite', desc: 'IR metrics · LLM-as-Judge · DeepEval'                      },
  '/ingest':    { label: 'Data Ingestion',   desc: 'Upload XLSX · rebuild Qdrant + BM25 index'                 },
}

interface HealthReady {
  status: string
  checks: Record<string, string>
}

export default function TopBar() {
  const { pathname } = useLocation()
  const toggle    = useUIStore(s => s.toggleSidebar)
  const collapsed = useUIStore(s => s.sidebarCollapsed)
  const meta = TITLES[pathname] ?? TITLES['/']

  // Poll /health/ready every 30 s to reflect real service status
  const { data: health } = useQuery<HealthReady>({
    queryKey: ['health-ready'],
    queryFn: async () => {
      const res = await fetch('/health/ready')
      return res.json()          // parse JSON even on 503
    },
    refetchInterval: 30_000,
    staleTime:       25_000,
    retry: false,
  })

  const qdrantOk: boolean | undefined = health
    ? health.checks?.qdrant === 'ok'
    : undefined

  return (
    <div className="topbar">
      {/* Sidebar toggle */}
      <button className="btn-ghost p-2" onClick={toggle} aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}>
        <Menu className="w-4 h-4" />
      </button>

      {/* Page title */}
      <motion.div
        key={pathname}
        initial={{ opacity: 0, y: -4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.18 }}
        className="flex-1 min-w-0"
      >
        <div className="flex items-baseline gap-3 flex-wrap">
          <h1 className="text-[15px] font-semibold" style={{ color: 'var(--text-primary)' }}>
            {meta.label}
          </h1>
          <span className="hidden md:block text-[12px]" style={{ color: 'var(--text-secondary)' }}>
            {meta.desc}
          </span>
        </div>
      </motion.div>

      {/* Live status indicators */}
      <div className="hidden md:flex items-center gap-4">
        <StatusPill label="Qdrant" ok={qdrantOk} />
      </div>
    </div>
  )
}

function StatusPill({ label, ok }: { label: string; ok?: boolean }) {
  // undefined = still loading → grey, true = up → teal, false = down → red
  const color =
    ok === undefined ? '#6B7280' :
    ok              ? '#23C6A8' :
                      '#F05A5A'
  const title =
    ok === undefined ? `${label}: checking…` :
    ok              ? `${label}: connected`  :
                      `${label}: unreachable`

  return (
    <div
      className="flex items-center gap-1.5 text-[11px]"
      style={{ color: 'var(--text-secondary)' }}
      title={title}
    >
      <Circle className="w-2 h-2 fill-current" style={{ color }} />
      {label}
    </div>
  )
}
