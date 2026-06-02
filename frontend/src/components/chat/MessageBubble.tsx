import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import { ThumbsUp, ThumbsDown } from 'lucide-react'
import { submitFeedback } from '../../api/feedbackApi'
import type { ChatMessage } from '../../store/chatStore'

interface Props {
  message: ChatMessage
  onActionClick: (action: string) => void
  sessionId: string | null
}

type ThumbState = 'idle' | 'down_open' | 'done_up' | 'done_down'

export function MessageBubble({ message, onActionClick, sessionId }: Props) {
  const isUser = message.role === 'user'

  // Only assistant messages that presented a fix get thumbs
  const showThumbs = !isUser && !!message.optionProgress && !message.isEscalated

  const [thumb, setThumb]       = useState<ThumbState>('idle')
  const [reason, setReason]     = useState('')
  const [submitting, setSubmit] = useState(false)

  async function handleThumbUp() {
    if (thumb !== 'idle' || !sessionId || !message.optionProgress) return
    setThumb('done_up')
    try {
      await submitFeedback(sessionId, message.optionProgress.current, 'positive')
    } catch { /* silent — UI already updated */ }
  }

  function handleThumbDown() {
    if (thumb !== 'idle') return
    setThumb('down_open')
  }

  async function handleReasonSubmit() {
    if (!sessionId || !message.optionProgress) return
    setSubmit(true)
    try {
      await submitFeedback(
        sessionId,
        message.optionProgress.current,
        'negative',
        reason.trim() || undefined,
      )
    } catch { /* silent */ }
    setThumb('done_down')
    setSubmit(false)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18 }}
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-3`}
    >
      <div className={`max-w-[78%] ${isUser ? '' : 'w-full max-w-[78%]'}`}>

        {/* Bubble */}
        <div
          className={`rounded-2xl px-4 py-3 ${isUser ? 'rounded-br-sm text-white' : 'rounded-bl-sm'}`}
          style={{
            background: isUser ? 'var(--accent-blue)' : 'rgba(255,255,255,0.06)',
            color: isUser ? '#fff' : 'var(--text-primary)',
            border: isUser ? 'none' : '1px solid var(--border)',
          }}
        >
          {/* Option progress pill */}
          {message.optionProgress && (
            <div
              className="text-[11px] font-mono mb-2 px-2 py-0.5 rounded-full inline-block"
              style={{ background: 'rgba(79,140,255,0.18)', color: 'var(--accent-blue)' }}
            >
              Fix {message.optionProgress.current} of {message.optionProgress.total}
            </div>
          )}

          {/* Message body */}
          {isUser ? (
            <p className="text-sm leading-relaxed">{message.content}</p>
          ) : (
            <div className="prose prose-invert prose-sm max-w-none text-sm leading-relaxed
                            [&_ol]:pl-4 [&_li]:mb-1 [&_strong]:text-white">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          )}

          {/* Escalation badge */}
          {message.isEscalated && message.escalationTicketId && (
            <div
              className="mt-3 flex items-center gap-2 text-xs px-3 py-1.5 rounded-lg"
              style={{ background: 'rgba(239,68,68,0.15)', color: '#f87171' }}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-red-400 flex-shrink-0" />
              <span>Escalated · Ticket: <code className="font-mono">{message.escalationTicketId}</code></span>
            </div>
          )}

          {/* Suggested action buttons */}
          {message.suggestedActions && message.suggestedActions.length > 0 && (
            <div className="flex flex-wrap gap-2 mt-3">
              {message.suggestedActions.map(action => (
                <button
                  key={action}
                  onClick={() => onActionClick(action)}
                  className="text-xs px-3 py-1.5 rounded-full transition-colors"
                  style={{
                    border: '1px solid var(--border)',
                    color: 'var(--text-secondary)',
                    background: 'transparent',
                  }}
                  onMouseEnter={e => ((e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.07)')}
                  onMouseLeave={e => ((e.currentTarget as HTMLElement).style.background = 'transparent')}
                >
                  {action}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* ── Thumbs row — ChatGPT style ──────────────────────────────── */}
        {showThumbs && (
          <div className="mt-1 ml-1">

            {/* Idle — small muted icons */}
            {thumb === 'idle' && (
              <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity"
                   style={{ opacity: 0.45 }}
                   onMouseEnter={e => (e.currentTarget as HTMLElement).style.opacity = '1'}
                   onMouseLeave={e => (e.currentTarget as HTMLElement).style.opacity = '0.45'}>
                <button
                  onClick={handleThumbUp}
                  title="This helped"
                  className="p-0.5 rounded transition-colors"
                  style={{ color: 'var(--text-secondary)', background: 'none', border: 'none', cursor: 'pointer' }}
                  onMouseEnter={e => (e.currentTarget as HTMLElement).style.color = '#23C6A8'}
                  onMouseLeave={e => (e.currentTarget as HTMLElement).style.color = 'var(--text-secondary)'}
                >
                  <ThumbsUp className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={handleThumbDown}
                  title="This didn't help"
                  className="p-0.5 rounded transition-colors"
                  style={{ color: 'var(--text-secondary)', background: 'none', border: 'none', cursor: 'pointer' }}
                  onMouseEnter={e => (e.currentTarget as HTMLElement).style.color = '#F05A5A'}
                  onMouseLeave={e => (e.currentTarget as HTMLElement).style.color = 'var(--text-secondary)'}
                >
                  <ThumbsDown className="w-3.5 h-3.5" />
                </button>
              </div>
            )}

            {/* Thumbs down → reason panel */}
            <AnimatePresence>
              {thumb === 'down_open' && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.18 }}
                  className="overflow-hidden"
                >
                  <div
                    className="mt-1 rounded-xl p-3"
                    style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border)' }}
                  >
                    <p className="text-[11px] mb-2" style={{ color: 'var(--text-secondary)' }}>
                      What went wrong? <span style={{ opacity: 0.5 }}>(optional)</span>
                    </p>
                    <textarea
                      autoFocus
                      value={reason}
                      onChange={e => setReason(e.target.value)}
                      placeholder="e.g. restarted the service but errors persisted…"
                      rows={2}
                      className="w-full resize-none rounded-lg px-3 py-2 text-[12px] focus:outline-none"
                      style={{
                        background: 'rgba(255,255,255,0.05)',
                        border: '1px solid var(--border)',
                        color: 'var(--text-primary)',
                      }}
                      onFocus={e => (e.currentTarget.style.borderColor = 'rgba(240,90,90,0.4)')}
                      onBlur={e => (e.currentTarget.style.borderColor = 'var(--border)')}
                    />
                    <div className="flex items-center gap-2 mt-2">
                      <button
                        onClick={handleReasonSubmit}
                        disabled={submitting}
                        className="text-[11px] px-3 py-1 rounded-lg font-medium transition-opacity disabled:opacity-40"
                        style={{ background: 'rgba(240,90,90,0.18)', color: '#F05A5A', border: '1px solid rgba(240,90,90,0.25)' }}
                      >
                        {submitting ? 'Sending…' : 'Send feedback'}
                      </button>
                      <button
                        onClick={() => setThumb('idle')}
                        className="text-[11px] px-2 py-1 rounded-lg"
                        style={{ color: 'var(--text-secondary)', background: 'none', border: 'none', cursor: 'pointer' }}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Done states */}
            {thumb === 'done_up' && (
              <div className="flex items-center gap-1 mt-0.5">
                <ThumbsUp className="w-3.5 h-3.5" style={{ color: '#23C6A8' }} />
                <span className="text-[11px]" style={{ color: '#23C6A8' }}>Thanks for the feedback</span>
              </div>
            )}
            {thumb === 'done_down' && (
              <div className="flex items-center gap-1 mt-0.5">
                <ThumbsDown className="w-3.5 h-3.5" style={{ color: '#F05A5A' }} />
                <span className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>Feedback sent</span>
              </div>
            )}
          </div>
        )}
      </div>
    </motion.div>
  )
}
