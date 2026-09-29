import { useEffect, useState } from 'react'
import { listRequests, updateRequest } from '../../api/client'

const STATUS_OPTS = [
  { value: '', label: 'Tất cả' },
  { value: 'pending', label: '🟡 Chờ xử lý' },
  { value: 'approved', label: '🟢 Đã duyệt' },
  { value: 'rejected', label: '🔴 Đã từ chối' },
]

function StatusBadge({ status }) {
  const map = { pending: '🟡 Chờ xử lý', approved: '🟢 Đã duyệt', rejected: '🔴 Đã từ chối' }
  return <span className={`badge badge-${status}`}>{map[status] ?? status}</span>
}

function RequestCard({ req, onUpdate }) {
  const [note, setNote] = useState('')
  const [acting, setActing] = useState(false)

  const handle = async (newStatus) => {
    setActing(true)
    try {
      await updateRequest(req.request_id, newStatus, note)
      onUpdate()
    } finally {
      setActing(false)
    }
  }

  return (
    <div className="request-card">
      <div className="request-card-header">
        <div>
          <div className="request-card-title">📄 {req.subject}</div>
          <div className="request-card-meta">
            👤 {req.username} &nbsp;·&nbsp; 🕒 {req.created_at?.slice(0, 16).replace('T', ' ')}
          </div>
        </div>
        <StatusBadge status={req.status} />
      </div>

      {req.description && (
        <div className="request-card-desc">📝 {req.description}</div>
      )}

      {req.teacher_note && (
        <div className="teacher-note">💬 Ghi chú GV: {req.teacher_note}</div>
      )}

      {req.status === 'pending' && (
        <div style={{ marginTop: 12 }}>
          <input
            className="form-input"
            placeholder="Ghi chú phản hồi (tùy chọn)..."
            value={note}
            onChange={e => setNote(e.target.value)}
            style={{ marginBottom: 10 }}
          />
          <div className="request-card-actions">
            <button
              className="btn btn-success btn-sm"
              onClick={() => handle('approved')}
              disabled={acting}
            >✅ Duyệt</button>
            <button
              className="btn btn-danger btn-sm"
              onClick={() => handle('rejected')}
              disabled={acting}
            >❌ Từ chối</button>
          </div>
        </div>
      )}
    </div>
  )
}

export default function TeacherRequests() {
  const [requests, setRequests] = useState([])
  const [filter, setFilter] = useState('')
  const [loading, setLoading] = useState(true)

  const load = () => {
    setLoading(true)
    listRequests(filter || undefined).then(setRequests).finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [filter])

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">📬 Yêu cầu tài liệu</h1>
        <p className="page-subtitle">Xem và xử lý yêu cầu từ học sinh</p>
      </div>

      <div style={{ display: 'flex', gap: 12, marginBottom: 20, alignItems: 'center' }}>
        <select className="form-select" style={{ maxWidth: 200 }} value={filter} onChange={e => setFilter(e.target.value)}>
          {STATUS_OPTS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
        <button className="btn btn-ghost btn-sm" onClick={load}>🔄 Làm mới</button>
        {!loading && <span style={{ fontSize: '0.85rem', color: 'var(--gray-500)' }}>
          {requests.length} yêu cầu
        </span>}
      </div>

      {loading ? (
        <div className="loading-center"><div className="spinner" /></div>
      ) : requests.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">📭</div>
          <div className="empty-state-title">Không có yêu cầu nào</div>
          <p>Học sinh chưa gửi yêu cầu tài liệu nào.</p>
        </div>
      ) : (
        requests.map(req => (
          <RequestCard key={req.request_id} req={req} onUpdate={load} />
        ))
      )}
    </div>
  )
}
