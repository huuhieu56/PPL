import { useEffect, useState, useRef } from 'react'
import { listDocuments, uploadDocuments, deleteDocument } from '../../api/client'

function FileTag({ name, onRemove }) {
  return (
    <span className="file-tag">
      📄 {name}
      <button onClick={onRemove} title="Bỏ file này">✕</button>
    </span>
  )
}

function StatusBadge({ status }) {
  return <span className={`badge badge-${status}`}>{status}</span>
}

export default function TeacherDocuments() {
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadMsg, setUploadMsg] = useState(null)   // {type, text}
  const [deleteConfirm, setDeleteConfirm] = useState(null)  // doc_id pending confirm

  // Upload form state
  const [course, setCourse] = useState('')
  const [sourceType, setSourceType] = useState('textbook')
  const [semester, setSemester] = useState('')
  const [files, setFiles] = useState([])
  const [dragover, setDragover] = useState(false)
  const fileInputRef = useRef()

  const load = () => {
    setLoading(true)
    listDocuments().then(setDocuments).finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  const addFiles = newFiles => {
    const arr = Array.from(newFiles)
    setFiles(prev => {
      const names = new Set(prev.map(f => f.name))
      return [...prev, ...arr.filter(f => !names.has(f.name))]
    })
  }

  const removeFile = idx => setFiles(prev => prev.filter((_, i) => i !== idx))

  const handleDrop = e => {
    e.preventDefault(); setDragover(false)
    addFiles(e.dataTransfer.files)
  }

  const handleUpload = async e => {
    e.preventDefault()
    if (!course.trim()) { setUploadMsg({ type: 'error', text: 'Vui lòng nhập tên môn học.' }); return }
    if (!files.length) { setUploadMsg({ type: 'error', text: 'Vui lòng chọn ít nhất 1 file.' }); return }

    const fd = new FormData()
    fd.append('course', course.trim())
    fd.append('source_type', sourceType)
    fd.append('semester', semester.trim())
    files.forEach(f => fd.append('files', f))

    setUploading(true); setUploadMsg(null); setUploadProgress(0)
    try {
      const res = await uploadDocuments(fd, setUploadProgress)
      setUploadMsg({ type: 'success', text: `✅ ${res.message}` })
      setFiles([]); setCourse(''); setSemester('')
      load()
    } catch (err) {
      setUploadMsg({ type: 'error', text: err.response?.data?.detail || 'Upload thất bại.' })
    } finally {
      setUploading(false)
    }
  }

  const handleDelete = async (docId) => {
    try {
      await deleteDocument(docId)
      setDeleteConfirm(null)
      load()
    } catch (err) {
      alert(err.response?.data?.detail || 'Xóa thất bại.')
    }
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">📚 Quản lý tài liệu</h1>
        <p className="page-subtitle">Tải lên, xem và xóa tài liệu học tập</p>
      </div>

      {/* ── Upload form ─────────────────────────────────────────── */}
      <div className="card" style={{ marginBottom: 24 }}>
        <h2 className="card-title">⬆️ Tải lên tài liệu mới</h2>
        <hr className="divider" />

        <form onSubmit={handleUpload}>
          <div className="form-row cols-3" style={{ marginBottom: 16 }}>
            <div className="form-group" style={{ margin: 0 }}>
              <label className="form-label">Môn học *</label>
              <input className="form-input" placeholder="VD: AI101" value={course} onChange={e => setCourse(e.target.value)} />
            </div>
            <div className="form-group" style={{ margin: 0 }}>
              <label className="form-label">Loại tài liệu</label>
              <select className="form-select" value={sourceType} onChange={e => setSourceType(e.target.value)}>
                <option value="textbook">Giáo trình</option>
                <option value="slide">Slide</option>
                <option value="exam">Đề thi</option>
              </select>
            </div>
            <div className="form-group" style={{ margin: 0 }}>
              <label className="form-label">Học kỳ</label>
              <input className="form-input" placeholder="VD: HK1-2024" value={semester} onChange={e => setSemester(e.target.value)} />
            </div>
          </div>

          {/* Drop zone */}
          <div
            className={`upload-zone${dragover ? ' dragover' : ''}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={e => { e.preventDefault(); setDragover(true) }}
            onDragLeave={() => setDragover(false)}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx,.pptx"
              multiple
              onChange={e => addFiles(e.target.files)}
            />
            <div className="upload-zone-icon">☁️</div>
            <div className="upload-zone-title">Kéo thả hoặc nhấn để chọn file</div>
            <div className="upload-zone-subtitle">Hỗ trợ PDF, DOCX, PPTX — tối đa 200 MB</div>
          </div>

          {files.length > 0 && (
            <div className="file-list">
              {files.map((f, i) => (
                <FileTag key={i} name={f.name} onRemove={() => removeFile(i)} />
              ))}
            </div>
          )}

          {uploadMsg && (
            <div className={`alert alert-${uploadMsg.type === 'success' ? 'success' : 'error'}`} style={{ marginTop: 12 }}>
              {uploadMsg.text}
            </div>
          )}

          {uploading && (
            <div style={{ marginTop: 12 }}>
              <div style={{ height: 8, background: 'var(--gray-200)', borderRadius: 4, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${uploadProgress}%`, background: 'var(--primary)', transition: 'width .3s' }} />
              </div>
              <p style={{ fontSize: '0.8rem', color: 'var(--gray-500)', marginTop: 4 }}>
                Đang trích xuất và lập chỉ mục... {uploadProgress}%
              </p>
            </div>
          )}

          <button
            type="submit"
            className="btn btn-primary"
            disabled={uploading}
            style={{ marginTop: 16 }}
          >
            {uploading ? '⏳ Đang xử lý...' : '🚀 Tải lên & lập chỉ mục'}
          </button>
        </form>
      </div>

      {/* ── Document list ────────────────────────────────────────── */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h2 className="card-title">📄 Tài liệu trong corpus</h2>
          <button className="btn btn-ghost btn-sm" onClick={load}>🔄 Làm mới</button>
        </div>

        {loading ? (
          <div className="loading-center"><div className="spinner" /></div>
        ) : documents.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">📭</div>
            <div className="empty-state-title">Chưa có tài liệu nào</div>
            <p>Tải lên tài liệu đầu tiên bằng form ở trên.</p>
          </div>
        ) : (
          <>
            <p style={{ fontSize: '0.85rem', color: 'var(--gray-500)', marginBottom: 12 }}>
              Tổng cộng <strong>{documents.length}</strong> tài liệu.
              Nhấn 🗑️ để xóa khỏi DB và ổ đĩa — nên lập chỉ mục lại sau khi xóa.
            </p>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Tên file</th>
                    <th>Môn học</th>
                    <th>Loại</th>
                    <th>Học kỳ</th>
                    <th>Trạng thái</th>
                    <th>Xóa</th>
                  </tr>
                </thead>
                <tbody>
                  {documents.map(doc => (
                    <tr key={doc.doc_id}>
                      <td><strong>{doc.filename || doc.doc_id}</strong></td>
                      <td>{doc.course || '-'}</td>
                      <td>{doc.source_type || '-'}</td>
                      <td>{doc.semester || '-'}</td>
                      <td><StatusBadge status={doc.status || 'processed'} /></td>
                      <td>
                        {deleteConfirm === doc.doc_id ? (
                          <div className="td-actions">
                            <button
                              className="btn btn-danger btn-sm"
                              onClick={() => handleDelete(doc.doc_id)}
                            >✅ Xác nhận</button>
                            <button
                              className="btn btn-ghost btn-sm"
                              onClick={() => setDeleteConfirm(null)}
                            >❌</button>
                          </div>
                        ) : (
                          <button
                            className="btn btn-ghost btn-sm btn-icon"
                            title="Xóa tài liệu"
                            onClick={() => setDeleteConfirm(doc.doc_id)}
                          >🗑️</button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
