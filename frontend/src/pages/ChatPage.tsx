import { useRef, useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, RefreshCw, MessageSquare } from 'lucide-react'
import { useChatStore } from '../store/chatStore'
import { MessageBubble } from '../components/chat/MessageBubble'

const PLACEHOLDER_QUERIES = [
  'Database connections are failing after the latest deployment',
  'VPN keeps disconnecting every 15 minutes',
  'Storage volume is at 95% capacity and alerts are firing',
]

export default function ChatPage() {
  const { messages, isLoading, error, sendMessage, resetSession, sessionId } = useChatStore()
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  const handleSend = async (text: string) => {
    const trimmed = text.trim()
    if (!trimmed || isLoading) return
    setInput('')
    await sendMessage(trimmed)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend(input)
    }
  }

  const isEmpty = messages.length === 0

  return (
    <div className="flex flex-col" style={{ height: 'calc(100vh - var(--topbar-height))' }}>
      {/* ── Header ── */}
      <div
        className="flex items-center justify-between px-6 py-4 flex-shrink-0"
        style={{ borderBottom: '1px solid var(--border)' }}
      >
        <div className="flex items-center gap-2">
          <MessageSquare className="w-5 h-5" style={{ color: 'var(--accent-blue)' }} />
          <h1 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>
            Guided Troubleshooting
          </h1>
          <span
            className="text-[11px] px-2 py-0.5 rounded-full font-mono"
            style={{ background: 'rgba(79,140,255,0.12)', color: 'var(--accent-blue)' }}
          >
            Chat
          </span>
        </div>
        <button
          onClick={resetSession}
          disabled={isEmpty}
          className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg transition-colors disabled:opacity-30"
          style={{ color: 'var(--text-secondary)', border: '1px solid var(--border)' }}
          onMouseEnter={e =>
            !isEmpty && ((e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.05)')
          }
          onMouseLeave={e =>
            ((e.currentTarget as HTMLElement).style.background = 'transparent')
          }
        >
          <RefreshCw className="w-3.5 h-3.5" />
          New session
        </button>
      </div>

      {/* ── Message list ── */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        {isEmpty && (
          <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
            <div
              className="w-14 h-14 rounded-2xl flex items-center justify-center"
              style={{ background: 'rgba(79,140,255,0.1)' }}
            >
              <MessageSquare className="w-7 h-7" style={{ color: 'var(--accent-blue)' }} />
            </div>
            <div>
              <p className="text-base font-medium mb-1" style={{ color: 'var(--text-primary)' }}>
                Describe your IT incident
              </p>
              <p className="text-sm max-w-sm" style={{ color: 'var(--text-secondary)' }}>
                I'll walk you through proven resolution steps one by one.
                If a fix doesn't work, I'll suggest the next option from our incident knowledge base.
              </p>
            </div>
            <div className="flex flex-col gap-2 w-full max-w-sm mt-2">
              {PLACEHOLDER_QUERIES.map(q => (
                <button
                  key={q}
                  onClick={() => handleSend(q)}
                  className="text-left text-xs px-4 py-2.5 rounded-xl transition-colors"
                  style={{
                    background: 'rgba(255,255,255,0.04)',
                    border: '1px solid var(--border)',
                    color: 'var(--text-secondary)',
                  }}
                  onMouseEnter={e =>
                    ((e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.07)')
                  }
                  onMouseLeave={e =>
                    ((e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.04)')
                  }
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        <AnimatePresence initial={false}>
          {messages.map(msg => (
            <MessageBubble
              key={msg.id}
              message={msg}
              onActionClick={text => handleSend(text)}
              sessionId={sessionId}
            />
          ))}
        </AnimatePresence>

        {/* Typing indicator */}
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex justify-start mb-3"
          >
            <div
              className="rounded-2xl rounded-bl-sm px-4 py-3"
              style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid var(--border)' }}
            >
              <span className="inline-flex gap-1 items-center">
                {[0, 1, 2].map(i => (
                  <motion.span
                    key={i}
                    className="w-1.5 h-1.5 rounded-full"
                    style={{ background: 'var(--text-secondary)' }}
                    animate={{ opacity: [0.3, 1, 0.3] }}
                    transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2 }}
                  />
                ))}
              </span>
            </div>
          </motion.div>
        )}

        {error && (
          <p className="text-sm text-center mb-3" style={{ color: '#f87171' }}>{error}</p>
        )}

        <div ref={bottomRef} />
      </div>

      {/* ── Input bar ── */}
      <div
        className="px-6 py-4 flex-shrink-0"
        style={{ borderTop: '1px solid var(--border)' }}
      >
        <div className="flex gap-3 items-end">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Describe your incident or reply to the assistant…"
            rows={2}
            className="flex-1 resize-none rounded-xl px-4 py-3 text-sm focus:outline-none"
            style={{
              background: 'rgba(255,255,255,0.05)',
              border: '1px solid var(--border)',
              color: 'var(--text-primary)',
              transition: 'border-color 0.15s',
            }}
            onFocus={e => ((e.currentTarget as HTMLElement).style.borderColor = 'var(--accent-blue)')}
            onBlur={e => ((e.currentTarget as HTMLElement).style.borderColor = 'var(--border)')}
          />
          <button
            onClick={() => handleSend(input)}
            disabled={!input.trim() || isLoading}
            className="h-10 w-10 rounded-xl flex items-center justify-center flex-shrink-0 transition-opacity disabled:opacity-30"
            style={{ background: 'var(--accent-blue)' }}
          >
            <Send className="w-4 h-4 text-white" />
          </button>
        </div>
        <p className="text-[11px] mt-2" style={{ color: 'var(--text-secondary)' }}>
          Shift+Enter for new line · Enter to send
        </p>
      </div>
    </div>
  )
}
