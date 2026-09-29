import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

// Attach JWT token to every request
api.interceptors.request.use(config => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// Auto-logout on 401
api.interceptors.response.use(
  res => res,
  err => {
    if (err.response?.status === 401) {
      localStorage.removeItem('token')
      localStorage.removeItem('user')
      window.location.href = '/'
    }
    return Promise.reject(err)
  }
)

// ── Auth ──────────────────────────────────────────────────────────────────────
export const login = (username, password) =>
  api.post('/auth/login', { username, password }).then(r => r.data)

export const getMe = () => api.get('/auth/me').then(r => r.data)

// ── Corpus ────────────────────────────────────────────────────────────────────
export const getActiveCorpus = () => api.get('/corpus/active').then(r => r.data)

// ── Documents ─────────────────────────────────────────────────────────────────
export const listDocuments = () => api.get('/documents').then(r => r.data)

export const uploadDocuments = (formData, onProgress) =>
  api.post('/documents/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: e => onProgress?.(Math.round((e.loaded * 100) / e.total)),
  }).then(r => r.data)

export const deleteDocument = (docId) =>
  api.delete(`/documents/${docId}`).then(r => r.data)

// ── Doc Requests ──────────────────────────────────────────────────────────────
export const listRequests = (filterStatus) =>
  api.get('/requests', { params: filterStatus ? { filter_status: filterStatus } : {} }).then(r => r.data)

export const createRequest = (subject, description) =>
  api.post('/requests', { subject, description }).then(r => r.data)

export const updateRequest = (requestId, status, teacherNote = '') =>
  api.patch(`/requests/${requestId}`, { status, teacher_note: teacherNote }).then(r => r.data)

// ── Chat ──────────────────────────────────────────────────────────────────────
export const listSessions = (corpusVersion = '') =>
  api.get('/chat/sessions', { params: { corpus_version: corpusVersion } }).then(r => r.data)

export const createSession = (corpusVersion = '') =>
  api.post('/chat/sessions', null, { params: { corpus_version: corpusVersion } }).then(r => r.data)

export const getMessages = (sessionId) =>
  api.get(`/chat/sessions/${sessionId}/messages`).then(r => r.data)

export const askQuestion = (sessionId, query, docIds = []) =>
  api.post('/chat/ask', { session_id: sessionId, query, doc_ids: docIds }).then(r => r.data)

export const saveFeedback = (messageId, useful, comment = '') =>
  api.post('/chat/feedback', { message_id: messageId, useful, comment }).then(r => r.data)

// ── RAG Configs ───────────────────────────────────────────────────────────────
export const listRagConfigs = () => api.get('/rag-configs').then(r => r.data)
