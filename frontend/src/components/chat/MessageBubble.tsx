import { motion } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import type { ChatMessage } from '../../store/chatStore'

interface Props {
  message: ChatMessage
  onActionClick: (action: string) => void
}

export function MessageBubble({ message, onActionClick }: Props) {
  const isUser = message.role === 'user'

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18 }}
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-3`}
    >
      <div
        className={`max-w-[78%] rounded-2xl px-4 py-3 ${
          isUser
            ? 'rounded-br-sm text-white'
            : 'rounded-bl-sm'
        }`}
        style={{
          background: isUser
            ? 'var(--accent-blue)'
            : 'rgba(255,255,255,0.06)',
          color: isUser ? '#fff' : 'var(--text-primary)',
          border: isUser ? 'none' : '1px solid var(--border)',
        }}
      >
        {/* Option progress pill */}
        {message.optionProgress && (
          <div
            className="text-[11px] font-mono mb-2 px-2 py-0.5 rounded-full inline-block"
            style={{
              background: 'rgba(79,140,255,0.18)',
              color: 'var(--accent-blue)',
            }}
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
                onMouseEnter={e =>
                  ((e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.07)')
                }
                onMouseLeave={e =>
                  ((e.currentTarget as HTMLElement).style.background = 'transparent')
                }
              >
                {action}
              </button>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  )
}
