import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import AppLayout from './layouts/AppLayout'
import SearchPage from './pages/SearchPage'
import TriagePage from './pages/TriagePage'
import AnalyticsPage from './pages/AnalyticsPage'
import IngestionPage from './pages/IngestionPage'
import ChatPage from './pages/ChatPage'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 30,
      retry: 1,
    },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<AppLayout />}>
            <Route path="/"          element={<SearchPage />} />
            <Route path="/triage"    element={<TriagePage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
            <Route path="/ingest"    element={<IngestionPage />} />
            <Route path="/chat"      element={<ChatPage />} />
            <Route path="*"          element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
