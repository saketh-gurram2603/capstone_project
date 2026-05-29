import { create } from 'zustand'
import { sendChatMessage } from '../api/chatApi'
import type { OptionProgress } from '../api/chatApi'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  optionProgress?: OptionProgress
  isEscalated?: boolean
  escalationTicketId?: string | null
  suggestedActions?: string[]
}

interface ChatStore {
  sessionId: string | null
  messages: ChatMessage[]
  isLoading: boolean
  error: string | null
  sendMessage: (text: string) => Promise<void>
  resetSession: () => void
}

export const useChatStore = create<ChatStore>((set, get) => ({
  sessionId: null,
  messages: [],
  isLoading: false,
  error: null,

  sendMessage: async (text: string) => {
    const { sessionId } = get()

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      timestamp: new Date(),
    }
    set(s => ({ messages: [...s.messages, userMsg], isLoading: true, error: null }))

    try {
      const response = await sendChatMessage({ session_id: sessionId, message: text })
      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: response.message,
        timestamp: new Date(),
        optionProgress: response.option_progress ?? undefined,
        isEscalated: response.is_escalated,
        escalationTicketId: response.escalation_ticket_id,
        suggestedActions: response.suggested_actions,
      }
      set(s => ({
        sessionId: response.session_id,
        messages: [...s.messages, assistantMsg],
        isLoading: false,
      }))
    } catch (err) {
      set({ isLoading: false, error: (err as Error).message })
    }
  },

  resetSession: () => set({ sessionId: null, messages: [], error: null, isLoading: false }),
}))
