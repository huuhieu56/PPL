import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getActiveCorpus } from '../../api/client'

export default function StudentDashboard() {
  const navigate = useNavigate()
  const [corpus, setCorpus] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getActiveCorpus().then(setCorpus).finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="loading-center"><div className="spinner" /></div>

  const docs = corpus?.documents ?? []
  const courses = [...new Set(docs.map(d => d.course).filter(Boolean))]

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">🎓 Không gian học tập</h1>
        <p className="page-subtitle">Hỏi đáp tài liệu học thuật bằng AI — có dẫn nguồn kiểm chứng</p>
      </div>

      {/* Stats */}
      {corpus?.active ? (
        <>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value">{docs.length}</div>
              <div className="stat-label">📄 Tài liệu có sẵn</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{courses.length}</div>
              <div className="stat-label">📖 Môn học</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{corpus.chunk_count ?? 0}</div>
              <div className="stat-label">🔍 Đoạn lập chỉ mục</div>
            </div>
          </div>

          {/* Document list */}
          {docs.length > 0 && (
            <div className="card" style={{ marginBottom: 24 }}>
              <h2 className="card-title" style={{ marginBottom: 12 }}>📚 Tài liệu hiện có</h2>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Tên tài liệu</th>
                      <th>Môn học</th>
                      <th>Loại</th>
                    </tr>
                  </thead>
                  <tbody>
                    {docs.map(doc => (
                      <tr key={doc.doc_id}>
                        <td>📄 {doc.filename || doc.doc_id}</td>
                        <td>{doc.course || '-'}</td>
                        <td>{doc.source_type || '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      ) : (
        <div className="alert alert-warning">
          ⚠️ Giáo viên chưa tải lên tài liệu nào. Bạn có thể gửi yêu cầu để giáo viên biết bạn cần gì.
        </div>
      )}

      <hr className="divider" />

      {/* Quick actions */}
      <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: 16 }}>🚀 Bắt đầu</h2>
      <div className="quick-links">
        <div className="quick-link-card" onClick={() => navigate('/chat')}>
          <span style={{ fontSize: '2rem' }}>💬</span>
          <span>Hỏi đáp tài liệu</span>
        </div>
        <div className="quick-link-card" onClick={() => navigate('/request')}>
          <span style={{ fontSize: '2rem' }}>📩</span>
          <span>Yêu cầu tài liệu mới</span>
        </div>
      </div>
    </div>
  )
}
