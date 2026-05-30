import { NavLink, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'
import {
  Search, GitBranch, BarChart2, Shield, ShieldCheck,
  ChevronLeft, ChevronRight, Activity, Upload, MessageSquare,
} from 'lucide-react'
import { useUIStore } from '../store/uiStore'
import { getFeedback } from '../api/feedbackApi'

const NAV = [
  {
    to: '/',
    label: 'Incident Search',
    icon: Search,
    badge: 'Core',
  },
  {
    to: '/triage',
    label: 'Agent Triage',
    icon: GitBranch,
    badge: 'L1→L3',
  },
  {
    to: '/analytics',
    label: 'Evaluation',
    icon: BarChart2,
    badge: null,
  },
  {
    to: '/ingest',
    label: 'Data Ingestion',
    icon: Upload,
    badge: null,
  },
  {
    to: '/chat',
    label: 'Chat Assistant',
    icon: MessageSquare
  },
  {
    to: '/admin',
    label: 'Feedback Review',
    icon: ShieldCheck,
    badge: null,
  },
]

export default function Sidebar() {
  const collapsed = useUIStore(s => s.sidebarCollapsed)
  const toggle    = useUIStore(s => s.toggleSidebar)
  const { pathname } = useLocation()

  // Live count of unreviewed feedback → red pill on the Feedback Review item
  const { data: fb } = useQuery({
    queryKey: ['feedback-pending'],
    queryFn: () => getFeedback(),
    refetchInterval: 5000,
  })
  const pendingFeedback = fb?.stats.pending ?? 0

  return (
    <motion.aside
      className="sidebar-rail select-none"
      animate={{ width: collapsed ? 72 : 260 }}
      transition={{ type: 'spring', stiffness: 380, damping: 34 }}
    >
      {/* ── Logo ──────────────────────────────────────────── */}
      <div className="flex items-center gap-3 py-4"
           style={{
             height: 'var(--topbar-height)',
             borderBottom: '1px solid var(--border)',
             padding: collapsed ? '0' : '0 1rem',
             justifyContent: collapsed ? 'center' : 'flex-start',
           }}>
        <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
             style={{ background: 'linear-gradient(135deg,#4F8CFF,#23C6A8)' }}>
          <Shield className="w-5 h-5 text-white" />
        </div>
        <AnimatePresence>
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -8 }}
              transition={{ duration: 0.15 }}
              className="overflow-hidden whitespace-nowrap"
            >
              <p className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>
                Incident KB
              </p>
              <p className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>
                AI Operations
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ── Nav ───────────────────────────────────────────── */}
      <nav className="flex-1 py-4 space-y-0.5 overflow-hidden" style={{ padding: collapsed ? '1rem 0.375rem' : '1rem 0.5rem' }}>
        <AnimatePresence>
          {!collapsed && (
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="section-label px-3 mb-3"
            >
              Workspace
            </motion.p>
          )}
        </AnimatePresence>

        {NAV.map(({ to, label, icon: Icon, badge }) => {
          const isActive = to === '/'
            ? pathname === '/'
            : pathname.startsWith(to)
          return (
            <NavLink key={to} to={to} end={to === '/'}>
              <motion.div
                whileHover={{ x: collapsed ? 0 : 2 }}
                transition={{ duration: 0.1 }}
                title={collapsed ? label : undefined}
                className="flex items-center rounded-xl cursor-pointer relative"
                style={{
                  gap: collapsed ? 0 : '0.75rem',
                  padding: collapsed ? '0.625rem 0' : '0.625rem 0.75rem',
                  justifyContent: collapsed ? 'center' : 'flex-start',
                  background: isActive ? 'rgba(79,140,255,0.12)' : 'transparent',
                  border: `1px solid ${isActive ? 'rgba(79,140,255,0.22)' : 'transparent'}`,
                  color: isActive ? 'var(--accent-blue)' : 'var(--text-secondary)',
                  transition: 'background 0.15s, border-color 0.15s, color 0.15s',
                }}
                onMouseEnter={e => {
                  if (!isActive) {
                    ;(e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.04)'
                    ;(e.currentTarget as HTMLElement).style.color = 'var(--text-primary)'
                  }
                }}
                onMouseLeave={e => {
                  if (!isActive) {
                    ;(e.currentTarget as HTMLElement).style.background = 'transparent'
                    ;(e.currentTarget as HTMLElement).style.color = 'var(--text-secondary)'
                  }
                }}
              >
                {isActive && !collapsed && (
                  <motion.div
                    layoutId="nav-indicator"
                    className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-5 rounded-r-full"
                    style={{ background: 'var(--accent-blue)' }}
                    transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                  />
                )}
                <Icon className="w-4 h-4 flex-shrink-0" />
                {collapsed && to === '/admin' && pendingFeedback > 0 && (
                  <span className="absolute top-1.5 right-2 w-2 h-2 rounded-full"
                        style={{ background: '#F05A5A', border: '1px solid var(--bg-secondary)' }} />
                )}
                <AnimatePresence>
                  {!collapsed && (
                    <motion.div
                      initial={{ opacity: 0, x: -6 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -6 }}
                      transition={{ duration: 0.13 }}
                      className="flex-1 flex items-center justify-between overflow-hidden"
                    >
                      <span className="text-sm font-medium whitespace-nowrap">{label}</span>
                      {to === '/admin' && pendingFeedback > 0 ? (
                        <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-full min-w-[18px] text-center"
                              style={{ background: 'rgba(240,90,90,0.18)', color: '#F05A5A',
                                       border: '1px solid rgba(240,90,90,0.3)' }}>
                          {pendingFeedback}
                        </span>
                      ) : badge ? (
                        <span className="mono text-[10px] px-1.5 py-0.5 rounded-md"
                              style={{ background: 'rgba(255,255,255,0.06)', color: 'var(--text-secondary)' }}>
                          {badge}
                        </span>
                      ) : null}
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            </NavLink>
          )
        })}
      </nav>

      {/* ── Collapse toggle ───────────────────────────────── */}
      <div style={{ borderTop: '1px solid var(--border)', padding: collapsed ? '0.75rem 0.375rem' : '0.75rem 0.5rem' }}>
        {!collapsed && (
          <div className="px-3 mb-2 flex items-center gap-1.5">
            <Activity className="w-3 h-3" style={{ color: 'var(--accent-teal)' }} />
            <span className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>
              v1.0.0 · development
            </span>
          </div>
        )}
        <button
          onClick={toggle}
          className="btn-ghost w-full justify-center"
          style={{ color: 'var(--text-secondary)' }}
        >
          {collapsed
            ? <ChevronRight className="w-4 h-4" />
            : <><ChevronLeft className="w-4 h-4" /><span className="text-xs">Collapse</span></>
          }
        </button>
      </div>
    </motion.aside>
  )
}
