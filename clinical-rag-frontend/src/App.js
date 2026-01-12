import React, { useState, useEffect, useRef } from 'react';
import { MessageSquare, Plus, Send, LogOut, User } from 'lucide-react';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  const [showAuthForm, setShowAuthForm] = useState('login');
  const [sessionId, setSessionId] = useState(() => generateUUID());
  const [chatHistory, setChatHistory] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [question, setQuestion] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [sessionsError, setSessionsError] = useState(null);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const chatEndRef = useRef(null);
  const abortControllerRef = useRef(null);

  // Auth form state
  const [authForm, setAuthForm] = useState({
    email: '',
    password: '',
    username: '',
    full_name: ''
  });

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      verifyToken(token);
    }
  }, []);

  useEffect(() => {
    if (isAuthenticated) {
      fetchSessions();
    }
  }, [isAuthenticated]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0;
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }

  async function verifyToken(token) {
    try {
      const resp = await fetch(`${API_URL}/auth/me`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (resp.ok) {
        const userData = await resp.json();
        setUser(userData);
        setIsAuthenticated(true);
      } else {
        localStorage.removeItem('token');
      }
    } catch (err) {
      console.error('Token verification failed:', err);
      localStorage.removeItem('token');
    }
  }

  async function handleLogin(e) {
    e.preventDefault();
    try {
      const resp = await fetch(`${API_URL}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: authForm.email,
          password: authForm.password
        })
      });

      if (resp.ok) {
        const data = await resp.json();
        localStorage.setItem('token', data.access_token);
        await verifyToken(data.access_token);
      } else {
        const error = await resp.json();
        alert(error.detail || 'Login failed');
      }
    } catch (err) {
      alert('Login failed: ' + err.message);
    }
  }

  async function handleRegister(e) {
    e.preventDefault();
    try {
      const resp = await fetch(`${API_URL}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: authForm.email,
          password: authForm.password,
          username: authForm.username,
          full_name: authForm.full_name
        })
      });

      if (resp.ok) {
        alert('Registration successful! Please login.');
        setShowAuthForm('login');
        setAuthForm({ email: '', password: '', username: '', full_name: '' });
      } else {
        const error = await resp.json();
        alert(error.detail || 'Registration failed');
      }
    } catch (err) {
      alert('Registration failed: ' + err.message);
    }
  }

  function handleLogout() {
    localStorage.removeItem('token');
    setIsAuthenticated(false);
    setUser(null);
    setSessions([]);
    setChatHistory([]);
    setShowUserMenu(false);
  }

  async function fetchSessions() {
    const token = localStorage.getItem('token');
    if (!token) return;

    setSessionsLoading(true);
    setSessionsError(null);
    try {
      const resp = await fetch(`${API_URL}/sessions`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (resp.ok) {
        const data = await resp.json();
        if (Array.isArray(data)) {
          const formattedSessions = data.map(s => ({
            ...s,
            id: String(s.id)
          }));
          setSessions(formattedSessions);
        }
      } else if (resp.status === 401) {
        handleLogout();
      } else {
        setSessionsError(`Error: ${resp.status}`);
      }
    } catch (err) {
      console.error('Failed to fetch sessions:', err);
      setSessionsError(err.message);
    } finally {
      setSessionsLoading(false);
    }
  }

  async function fetchMessages(sid) {
    const token = localStorage.getItem('token');
    if (!token) return [];

    try {
      const resp = await fetch(`${API_URL}/sessions/${sid}/messages`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (resp.ok) {
        return await resp.json();
      } else if (resp.status === 401) {
        handleLogout();
      }
    } catch (err) {
      console.error('Failed to fetch messages:', err);
    }
    return [];
  }

  async function handleSendMessage() {
    if (!question.trim() || isStreaming) return;

    const token = localStorage.getItem('token');
    if (!token) {
      alert('Please login first');
      return;
    }

    const userQuestion = question.trim();
    setQuestion('');
    setChatHistory(prev => [...prev, { role: 'user', content: userQuestion }]);
    setIsStreaming(true);

    try {
      abortControllerRef.current = new AbortController();
      const response = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          question: userQuestion,
          session_id: sessionId
        }),
        signal: abortControllerRef.current.signal
      });

      if (response.status === 401) {
        handleLogout();
        throw new Error('Authentication required');
      }

      if (!response.ok) throw new Error('Network response was not ok');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let answer = '';

      setChatHistory(prev => [...prev, { role: 'assistant', content: '' }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        answer += chunk;

        setChatHistory(prev => {
          const newHistory = [...prev];
          newHistory[newHistory.length - 1] = { role: 'assistant', content: answer };
          return newHistory;
        });
      }

      await fetchSessions();
    } catch (err) {
      if (err.name !== 'AbortError') {
        setChatHistory(prev => [...prev, {
          role: 'assistant',
          content: `⚠️ Error: ${err.message}`
        }]);
      }
    } finally {
      setIsStreaming(false);
      abortControllerRef.current = null;
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  }

  function handleNewChat() {
    setSessionId(generateUUID());
    setChatHistory([]);
  }

  async function handleSelectSession(sid) {
    setSessionId(sid);
    const messages = await fetchMessages(sid);
    setChatHistory(messages);
  }

  if (!isAuthenticated) {
    return (
      <div className="auth-container">
        <div className="auth-box">
          <h1 className="auth-title">🩺 Clinical RAG Assistant</h1>
          
          <div className="auth-tabs">
            <button
              className={`auth-tab ${showAuthForm === 'login' ? 'active' : ''}`}
              onClick={() => setShowAuthForm('login')}
            >
              Login
            </button>
            <button
              className={`auth-tab ${showAuthForm === 'register' ? 'active' : ''}`}
              onClick={() => setShowAuthForm('register')}
            >
              Register
            </button>
          </div>

          {showAuthForm === 'login' ? (
            <div className="auth-form">
              <input
                type="email"
                placeholder="Email"
                value={authForm.email}
                onChange={(e) => setAuthForm({...authForm, email: e.target.value})}
                className="auth-input"
              />
              <input
                type="password"
                placeholder="Password"
                value={authForm.password}
                onChange={(e) => setAuthForm({...authForm, password: e.target.value})}
                className="auth-input"
              />
              <button onClick={handleLogin} className="auth-button">
                Login
              </button>
            </div>
          ) : (
            <div className="auth-form">
              <input
                type="email"
                placeholder="Email"
                value={authForm.email}
                onChange={(e) => setAuthForm({...authForm, email: e.target.value})}
                className="auth-input"
              />
              <input
                type="text"
                placeholder="Username"
                value={authForm.username}
                onChange={(e) => setAuthForm({...authForm, username: e.target.value})}
                className="auth-input"
              />
              <input
                type="text"
                placeholder="Full Name (optional)"
                value={authForm.full_name}
                onChange={(e) => setAuthForm({...authForm, full_name: e.target.value})}
                className="auth-input"
              />
              <input
                type="password"
                placeholder="Password"
                value={authForm.password}
                onChange={(e) => setAuthForm({...authForm, password: e.target.value})}
                className="auth-input"
              />
              <button onClick={handleRegister} className="auth-button">
                Register
              </button>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="sidebar-header">
          <button onClick={handleNewChat} className="new-chat-btn">
            <Plus size={18} />
            <span>New Chat</span>
          </button>
        </div>

        <div className="sessions-list">
          <h2 className="sessions-title">Chats</h2>
          
          {sessionsLoading && (
            <div className="sessions-loading">Loading sessions...</div>
          )}
          
          {sessionsError && (
            <div className="sessions-error">
              Error: {sessionsError}
            </div>
          )}
          
          {!sessionsLoading && sessions.length === 0 && (
            <div className="sessions-empty">No chats yet</div>
          )}
          
          {sessions.map(session => (
            <button
              key={session.id}
              onClick={() => handleSelectSession(session.id)}
              className={`session-item ${session.id === sessionId ? 'active' : ''}`}
            >
              <MessageSquare size={16} className="session-icon" />
              <span className="session-title">{session.title}</span>
            </button>
          ))}
        </div>

        {/* User Profile at Bottom */}
        <div className="user-profile">
          <button 
            className="user-profile-button"
            onClick={() => setShowUserMenu(!showUserMenu)}
          >
            <div className="user-avatar">
              <User size={20} />
            </div>
            <div className="user-info">
              <div className="user-name">{user?.full_name || user?.username}</div>
              <div className="user-email">{user?.email}</div>
            </div>
          </button>
          
          {showUserMenu && (
            <div className="user-menu">
              <button onClick={handleLogout} className="user-menu-item">
                <LogOut size={16} />
                <span>Logout</span>
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        {/* Header */}
        <div className="header">
          <h1 className="app-title">
            🩺 Clinical RAG Assistant
          </h1>
        </div>

        {/* Chat Area */}
        <div className="chat-area">
          {chatHistory.map((msg, idx) => (
            <div
              key={idx}
              className={`message ${msg.role === 'user' ? 'message-user' : 'message-assistant'}`}
            >
              <div className="message-bubble">
                <div style={{ overflowX: 'auto', maxWidth: '100%' }}>
                  {(msg.content || '').split('\n').map((line, i) => {
                    // Check if line is a bullet point
                    const isBullet = line.trim().startsWith('•') || 
                                     line.trim().startsWith('-') || 
                                     line.trim().match(/^\d+\./);
                    
                    return (
                      <p key={i} className={`message-text ${isBullet ? 'bullet-point' : ''}`}>
                        {line}
                      </p>
                    );
                  })}
                </div>
              </div>
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>

        {/* Input Area */}
        <div className="input-area">
          <div className="input-container">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a clinical / medical question"
              disabled={isStreaming}
              className="input-field"
            />
            <button
              onClick={handleSendMessage}
              disabled={isStreaming || !question.trim()}
              className="send-btn"
            >
              <Send size={18} />
              <span>Send</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;