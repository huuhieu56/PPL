import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './contexts/AuthContext'

import LoginPage from './pages/Login'
import TeacherLayout from './pages/teacher/Layout'
import TeacherDashboard from './pages/teacher/Dashboard'
import TeacherDocuments from './pages/teacher/Documents'
import TeacherRequests from './pages/teacher/Requests'
import StudentLayout from './pages/student/Layout'
import StudentDashboard from './pages/student/Dashboard'
import StudentChat from './pages/student/Chat'
import StudentRequestDoc from './pages/student/RequestDoc'

function ProtectedRoutes() {
  const { user, loading } = useAuth()

  if (loading) {
    return (
      <div className="loading-center" style={{ minHeight: '100vh' }}>
        <div className="spinner" />
      </div>
    )
  }

  if (!user) return <Navigate to="/login" replace />

  if (user.role === 'teacher') {
    return (
      <Routes>
        <Route path="/" element={<TeacherLayout />}>
          <Route index element={<TeacherDashboard />} />
          <Route path="documents" element={<TeacherDocuments />} />
          <Route path="requests" element={<TeacherRequests />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    )
  }

  // student
  return (
    <Routes>
      <Route path="/" element={<StudentLayout />}>
        <Route index element={<StudentDashboard />} />
        <Route path="chat" element={<StudentChat />} />
        <Route path="request" element={<StudentRequestDoc />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/*" element={<ProtectedRoutes />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}
