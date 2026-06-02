import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import AppLayout from './layouts/AppLayout'
import SearchPage from './pages/SearchPage'
import TriagePage from './pages/TriagePage'
import AnalyticsPage from './pages/AnalyticsPage'
import IngestionPage from './pages/IngestionPage'
import ChatPage from './pages/ChatPage'
import AdminPage from './pages/AdminPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route path="/"          element={<SearchPage />} />
          <Route path="/triage"    element={<TriagePage />} />
          <Route path="/analytics" element={<AnalyticsPage />} />
          <Route path="/ingest"    element={<IngestionPage />} />
          <Route path="/chat"      element={<ChatPage />} />
          <Route path="/admin"     element={<AdminPage />} />
          <Route path="*"          element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
