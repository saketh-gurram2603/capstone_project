import { Outlet } from 'react-router-dom'
import { motion } from 'framer-motion'
import Sidebar from './Sidebar'
import TopBar from './TopBar'
import { useUIStore } from '../store/uiStore'

export default function AppLayout() {
  const collapsed = useUIStore(s => s.sidebarCollapsed)
  const marginLeft = collapsed ? 72 : 260

  return (
    <div className="app-layout">
      <Sidebar />
      <motion.div
        className="content-area"
        animate={{ marginLeft }}
        transition={{ type: 'spring', stiffness: 380, damping: 34 }}
      >
        <TopBar />
        <main className="page-content">
          <Outlet />
        </main>
      </motion.div>
    </div>
  )
}
