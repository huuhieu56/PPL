import { useEffect, useRef, useState, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import {
  getActiveCorpus, listSessions, createSession, getMessages, askQuestion, saveFeedback,
} from '../../api/client'

function ThinkingDots() {
  return (
    <div style={{ display: 'flex', gap: 4, padding: '12px 16px' }}>
      {[0, 1, 2].map(i => (
        <div key={i} className="dot" style={{ animationDelay: `${i * 0.15}s` }} />
      ))}
    </div>
  )
}

function CitationList({ citations, docNames }) {
  const [open, setOpen] = useState({})
  if (!citations?.length) return null
  return (
    <div className="citations">
      {citations.map((c, i) => {
        const name = docNames[c.doc_id] ?? c.doc_id
        const label = `[${c.number ?? i + 1}] ${name} — trang ${c.page}`
        return (
          <details key={i} className="citation-item" open={open[i]}>
            <summary onClick={() => setOpen(p => ({ ...p, [i]: !p[i] }))}>{label}</summary>
            <div className="citation-text">{c.text}</div>
          </details>
        )
      })}
    </div>
  )
}

function FeedbackBar({ messageId, onFeedback }) {
  const [sent, setSent] = useState(false)
  if (sent) return <p style={{ fontSize: '0.75rem', color: 'var(--gray-400)', marginTop: 6 }}>✓ Cảm ơn phản hồi!</p>
  return (
    <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
      <span style={{ fontSize: '0.75rem', color: 'var(--gray-400)' }}>Hữu ích?</span>
      {[true, false].map(v => (
        <button
          key={String(v)}
          className="btn btn-ghost btn-sm"
          style={{ padding: '2px 8px', fontSize: '0.75rem' }}
          onClick={async () => {
            await saveFeedback(messageId, v)
            setSent(true)
            onFeedback?.(messageId, v)
          }}
        >{v ? '👍' : '👎'}</button>
      ))}
    </div>
  )
}

export default function StudentChat() {
  const [corpus, setCorpus] = useState(null)
  const [sessions, setSessions] = useState([])
  const [sessionId, setSessionId] = useState(null)
  const [messages, setMessages] = useState([])
  const [thinking, setThinking] = useState(false)
  const [query, setQuery] = useState('')
  const [docNames, setDocNames] = useState({})
  const [error, setError] = useState('')
  const bottomRef = useRef()
  const textareaRef = useRef()

  // Load corpus & doc names
  useEffect(() => {
    getActiveCorpus().then(c => {
      setCorpus(c)
      if (c?.documents) {
        const names = {}
        c.documents.forEach(d => { names[d.doc_id] = d.filename })
        setDocNames(names)
      }
    })
  }, [])

  // Load / create session
  const initSession = useCallback(async (cv) => {
    const slist = await listSessions(cv)
    if (slist.length) {
      setSessions(slist)
      const sid = slist[0].session_id
      setSessionId(sid)
      const turns = await getMessages(sid)
      setMessages(turns.map(t => ([
        { role: 'user', text: t.query },
        { role: 'assistant', text: t.answer, citations: t.citations, messageId: t.message_id },
      ])).flat())
    } else {
      const { session_id } = await createSession(cv)
      setSessions([{ session_id }])
      setSessionId(session_id)
      setMessages([])
    }
  }, [])

  useEffect(() => {
    if (corpus?.version_id) initSession(corpus.version_id)
  }, [corpus?.version_id])

  // Switch session
  const switchSession = async (sid) => {
    setSessionId(sid)
    const turns = await getMessages(sid)
    setMessages(turns.map(t => ([
      { role: 'user', text: t.query },
      { role: 'assistant', text: t.answer, citations: t.citations, messageId: t.message_id },
    ])).flat())
  }

  const newSession = async () => {
    const { session_id } = await createSession(corpus?.version_id ?? '')
    setSessions(prev => [{ session_id }, ...prev])
    setSessionId(session_id)
    setMessages([])
  }

  // Auto-scroll
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, thinking])

  const handleSend = async () => {
    const q = query.trim()
    if (!q || !sessionId || thinking) return
    if (!corpus?.active) { setError('Chưa có corpus. Giáo viên cần tải tài liệu trước.'); return }
    setQuery('')
    setError('')
    setMessages(prev => [...prev, { role: 'user', text: q }])
    setThinking(true)
    try {
      const res = await askQuestion(sessionId, q)
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: res.answer,
        citations: res.citations,
        messageId: res.message_id,
        latency: res.latency_ms,
      }])
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', text: '⚠️ ' + (err.response?.data?.detail || 'Lỗi hệ thống'), isError: true }])
    } finally {
      setThinking(false)
    }
  }

  const handleKeyDown = e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
  }

  if (!corpus) return <div className="loading-center"><div className="spinner" /></div>

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      {/* ── Session sidebar ─────────────── */}
      <div style={{ width: 200, background: 'var(--gray-900)', color: 'white', display: 'flex', flexDirection: 'column', flexShrink: 0 }}>
        <div style={{ padding: '16px 14px', borderBottom: '1px solid var(--gray-700)' }}>
          <div style={{ fontWeight: 600, fontSize: '0.85rem', marginBottom: 8 }}>💬 Phiên chat</div>
          <button className="btn btn-primary btn-sm btn-full" onClick={newSession}>+ Cuộc trò chuyện mới</button>
        </div>
        <div style={{ flex: 1, overflowY: 'auto', padding: '8px 0' }}>
          {sessions.map(s => (
            <button
              key={s.session_id}
              className={`nav-item${s.session_id === sessionId ? ' active' : ''}`}
              onClick={() => switchSession(s.session_id)}
              style={{ fontSize: '0.8rem' }}
            >
              💬 Phiên {s.session_id.slice(0, 8)}
            </button>
          ))}
        </div>
      </div>

      {/* ── Chat area ───────────────────── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Header */}
        <div style={{ padding: '14px 24px', background: 'white', borderBottom: '1px solid var(--gray-200)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <div style={{ fontWeight: 600 }}>💬 Hỏi đáp tài liệu</div>
            {corpus?.active && (
              <div style={{ fontSize: '0.78rem', color: 'var(--gray-400)' }}>
                Corpus: {corpus.version_id?.slice(0, 14)}... · {corpus.chunk_count} chunks
              </div>
            )}
          </div>
          {!corpus?.active && (
            <span className="badge badge-pending">Chưa có corpus</span>
          )}
        </div>

        {error && <div className="alert alert-error" style={{ margin: '12px 24px 0' }}>{error}</div>}

        {/* Messages */}
        <div className="chat-messages">
          {messages.length === 0 && !thinking && (
            <div className="empty-state">
              <div className="empty-state-icon">💬</div>
              <div className="empty-state-title">Bắt đầu hỏi đáp</div>
              <p>Nhập câu hỏi về tài liệu học tập phía dưới.</p>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`chat-message ${msg.role}`}>
              <div className={`chat-avatar ${msg.role === 'user' ? 'avatar-user' : 'avatar-bot'}`}>
                {msg.role === 'user' ? '👤' : '🤖'}
              </div>
              <div>
                <div className={`chat-bubble ${msg.role === 'user' ? 'bubble-user' : 'bubble-bot'}`}>
                  <ReactMarkdown>{msg.text}</ReactMarkdown>
                </div>
                {msg.role === 'assistant' && !msg.isError && (
                  <>
                    {msg.latency && (
                      <p style={{ fontSize: '0.72rem', color: 'var(--gray-400)', marginTop: 4 }}>
                        ⏱ {Math.round(msg.latency)} ms
                      </p>
                    )}
                    <CitationList citations={msg.citations} docNames={docNames} />
                    {msg.messageId && <FeedbackBar messageId={msg.messageId} />}
                  </>
                )}
              </div>
            </div>
          ))}

          {thinking && (
            <div className="chat-message">
              <div className="chat-avatar avatar-bot">🤖</div>
              <div className="chat-bubble bubble-bot"><ThinkingDots /></div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="chat-input-area">
          <div className="chat-input-row">
            <textarea
              ref={textareaRef}
              className="chat-input"
              rows={1}
              placeholder={corpus?.active ? 'Nhập câu hỏi về tài liệu... (Enter để gửi, Shift+Enter để xuống dòng)' : 'Chưa có corpus. Giáo viên cần tải tài liệu trước.'}
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={!corpus?.active || thinking}
            />
            <button
              className="btn btn-primary"
              onClick={handleSend}
              disabled={!query.trim() || !corpus?.active || thinking}
            >
              {thinking ? '⏳' : '📤'}
            </button>
          </div>
          <p style={{ fontSize: '0.72rem', color: 'var(--gray-400)', marginTop: 6 }}>
            Câu trả lời dựa trên học liệu đã nạp — luôn kiểm tra dẫn nguồn.
          </p>
        </div>
      </div>
    </div>
  )
}
