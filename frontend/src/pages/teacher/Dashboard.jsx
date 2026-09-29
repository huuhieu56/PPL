import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getActiveCorpus, listRequests } from '../../api/client'

export default function TeacherDashboard() {
  const navigate = useNavigate()
  const [corpus, setCorpus] = useState(null)
  const [pending, setPending] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([getActiveCorpus(), listRequests('pending')])
      .then(([c, r]) => { setCorpus(c); setPending(r) })
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="loading-center"><div className="spinner" /></div>

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">👩‍🏫 Bảng điều khiển</h1>
        <p className="page-subtitle">Tổng quan hệ thống học liệu RAG</p>
      </div>

      {/* Stats */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-value">{corpus?.documents?.length ?? 0}</div>
          <div className="stat-label">📄 Tài liệu đã nạp</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{corpus?.chunk_count ?? 0}</div>
          <div className="stat-label">🗂️ Chunks trong corpus</div>
        </div>
        <div className="stat-card">
          <div className="stat-value" style={{ color: pending.length ? 'var(--warning)' : 'var(--success)' }}>
            {pending.length}
          </div>
          <div className="stat-label">📬 Yêu cầu đang chờ</div>
        </div>
      </div>

      {/* Corpus status */}
      {corpus?.active ? (
        <div className="alert alert-success">
          ✅ Corpus đang hoạt động: <strong>{corpus.version_id}</strong>
          {corpus.chunk_count ? ` — ${corpus.chunk_count} chunks` : ''}
        </div>
      ) : (
        <div className="alert alert-warning">
          ⚠️ Chưa có corpus nào. Hãy vào <strong>Quản lý tài liệu</strong> để tải lên tài liệu đầu tiên.
        </div>
      )}

      {/* Pending requests preview */}
      {pending.length > 0 && (
        <>
          <hr className="divider" />
          <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: 12 }}>
            🔔 Yêu cầu tài liệu đang chờ xử lý
          </h2>
          {pending.slice(0, 4).map(req => (
            <div key={req.request_id} className="alert alert-info" style={{ marginBottom: 8 }}>
              📩 <strong>{req.username}</strong> yêu cầu: {req.subject}
              <span style={{ marginLeft: 8, fontSize: '0.78rem', color: 'var(--gray-500)' }}>
                {req.created_at?.slice(0, 10)}
              </span>
            </div>
          ))}
          {pending.length > 4 && (
            <p style={{ fontSize: '0.85rem', color: 'var(--gray-500)' }}>
              ... và {pending.length - 4} yêu cầu khác
            </p>
          )}
        </>
      )}

      <hr className="divider" />

      {/* Quick links */}
      <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: 16 }}>🔗 Truy cập nhanh</h2>
      <div className="quick-links">
        <div className="quick-link-card" onClick={() => navigate('/documents')}>
          <span style={{ fontSize: '2rem' }}>📚</span>
          <span>Quản lý tài liệu</span>
        </div>
        <div className="quick-link-card" onClick={() => navigate('/requests')}>
          <span style={{ fontSize: '2rem' }}>📬</span>
          <span>Xem yêu cầu</span>
        </div>
      </div>
    </div>
  )
}
