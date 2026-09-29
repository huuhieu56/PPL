import { useEffect, useState } from 'react'
import { createRequest, listRequests } from '../../api/client'
import { useAuth } from '../../contexts/AuthContext'

function StatusBadge({ status }) {
  const map = { pending: '🟡 Đang chờ', approved: '🟢 Đã duyệt', rejected: '🔴 Từ chối' }
  return <span className={`badge badge-${status}`}>{map[status] ?? status}</span>
}

export default function StudentRequestDoc() {
  const { user } = useAuth()
  const [subject, setSubject] = useState('')
  const [description, setDescription] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [msg, setMsg] = useState(null)
  const [myRequests, setMyRequests] = useState([])
  const [loading, setLoading] = useState(true)

  const loadMyRequests = () => {
    listRequests()
      .then(all => setMyRequests(all.filter(r => r.username === user?.username)))
      .finally(() => setLoading(false))
  }

  useEffect(() => { loadMyRequests() }, [])

  const handleSubmit = async e => {
    e.preventDefault()
    if (!subject.trim()) { setMsg({ type: 'error', text: 'Vui lòng nhập tên tài liệu.' }); return }
    setSubmitting(true); setMsg(null)
    try {
      await createRequest(subject.trim(), description.trim())
      setMsg({ type: 'success', text: '✅ Đã gửi yêu cầu! Giáo viên sẽ phản hồi sớm.' })
      setSubject(''); setDescription('')
      loadMyRequests()
    } catch (err) {
      setMsg({ type: 'error', text: err.response?.data?.detail || 'Gửi thất bại.' })
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">📩 Yêu cầu tài liệu</h1>
        <p className="page-subtitle">Gửi yêu cầu để giáo viên tải lên tài liệu bạn cần</p>
      </div>

      {/* Form */}
      <div className="card" style={{ maxWidth: 600, marginBottom: 32 }}>
        <h2 className="card-title" style={{ marginBottom: 16 }}>📝 Gửi yêu cầu mới</h2>

        {msg && (
          <div className={`alert alert-${msg.type === 'success' ? 'success' : 'error'}`}>
            {msg.text}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label">Môn học / Tên tài liệu *</label>
            <input
              className="form-input"
              placeholder="VD: Giáo trình Lập trình Python"
              value={subject}
              onChange={e => setSubject(e.target.value)}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Mô tả chi tiết (tùy chọn)</label>
            <textarea
              className="form-textarea"
              placeholder="Mô tả nội dung, tác giả, năm xuất bản nếu biết..."
              value={description}
              onChange={e => setDescription(e.target.value)}
            />
          </div>
          <button type="submit" className="btn btn-primary" disabled={submitting}>
            {submitting ? '⏳ Đang gửi...' : '📤 Gửi yêu cầu'}
          </button>
        </form>
      </div>

      {/* My requests */}
      <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: 16 }}>📋 Yêu cầu của tôi</h2>

      {loading ? (
        <div className="loading-center"><div className="spinner" /></div>
      ) : myRequests.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">📭</div>
          <div className="empty-state-title">Chưa có yêu cầu nào</div>
          <p>Điền form ở trên để gửi yêu cầu đầu tiên.</p>
        </div>
      ) : (
        myRequests.map(req => (
          <div key={req.request_id} className="request-card">
            <div className="request-card-header">
              <div>
                <div className="request-card-title">📄 {req.subject}</div>
                <div className="request-card-meta">
                  🕒 {req.created_at?.slice(0, 16).replace('T', ' ')}
                </div>
              </div>
              <StatusBadge status={req.status} />
            </div>
            {req.description && (
              <div className="request-card-desc">📝 {req.description}</div>
            )}
            {req.teacher_note && (
              <div className="teacher-note">💬 Phản hồi giáo viên: {req.teacher_note}</div>
            )}
          </div>
        ))
      )}
    </div>
  )
}
