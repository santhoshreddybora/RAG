# Clinical RAG – Clinical Question Answering with Hybrid Retrieval

A production-ready **Retrieval-Augmented Generation (RAG)** system for **clinical and healthcare question answering** with modern web interface, user authentication, and advanced RAG techniques.

---

## Key Features

### RAG Pipeline
- **End-to-end document processing**
  - PDF / DOCX / TXT ingestion with optional OCR
  - Text cleaning, chunking, and deduplication
  - Hybrid retrieval combining dense (vector) + sparse (BM25) search
  
### Advanced Retrieval
- **Dense Retrieval**: Sentence Transformer embeddings via EURI API
- **Lexical Retrieval**: BM25 (rank-bm25) for keyword matching
- **Hybrid Fusion**: RRF (Reciprocal Rank Fusion) merging
- **Reranking**: Cross-encoder reranking for precision
- **Semantic Caching**: Fast responses for repeated questions

### LLM Generation
- **EURI API Integration**: GPT-4.1-nano for responses
- **Context-Grounded**: Answers strictly from retrieved documents
- **Streaming Responses**: Real-time token-by-token output
- **Smart Formatting**: Automatic bullet points and table rendering

### User Management
- **JWT Authentication**: Secure token-based auth
- **User Registration/Login**: Email-based accounts
- **Session Management**: Persistent chat history per user
- **Auto-login**: Token verification on page load

### Modern UI/UX
- **React Frontend**: Fast, responsive single-page application
- **Real-time Streaming**: Live AI responses as they generate
- **Smart Formatting**: 
  - Bullet points for lists
  - HTML tables for tabular data
  - Mixed content support
- **Chat History**: Browse and resume previous conversations
- **Dark Theme**: Modern, professional interface
- **Loading States**: Smooth animations and feedback

### Evaluation & Tracking
- **Metrics**: Recall, Precision, Hallucination Rate, Faithfulness
- **MLflow Integration**: Experiment tracking and logging
- **Evaluation Dataset**: Test set for continuous improvement

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     React Frontend (Port 3000)              │
│  - User Authentication (JWT)                                │
│  - Chat Interface with Session Management                   │
│  - Real-time Streaming Responses                            │
│  - Smart Formatting (Tables, Bullets)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP/REST
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  FastAPI Backend (Port 8000)                │
│  - JWT Authentication & User Management                     │
│  - Session & Chat History Management                        │
│  - Hybrid Retrieval (BM25 + Vector Search)                  │
│  - LLM Generation with Streaming                            │
│  - Semantic Caching                                         │
└──────┬────────────┬─────────────┬────────────┬─────────────┘
       │            │             │            │
       │            │             │            │
   ┌───▼───┐   ┌───▼────┐   ┌───▼─────┐  ┌──▼──────┐
   │Pinecone│  │  BM25  │   │PostgreSQL│ │  EURI   │
   │Vector  │  | Index  │   │Users/Chat│ │   API   │
   │  DB    │  │  .pkl  │   │ Sessions │ │Embedding│
   └───────┘   └────────┘   └──────────┘ │   LLM   │
                                         └─────────┘
```

---

## Project Structure

```
clinical-rag/
├── app/                           # Backend (Python/FastAPI)
│   ├── main.py                    # FastAPI application entry
│   │
│   ├── auth/                      # Authentication module
│   │   ├── __init__.py
│   │   └── auth_utils.py          # JWT creation, verification, password hashing
│   │
│   ├── routers/                   # API route modules
│   │   ├── __init__.py
│   │   └── auth.py                # Auth endpoints (/register, /login, /me)
│   │
│   ├── db/                        # Database
│   │   ├── __init__.py
│   │   ├── database.py            # SQLAlchemy async engine
│   │   └── models.py              # User, ChatSession, ChatMessage, ChatSummary
│   │
│   ├── retrieval/                 # RAG retrieval components
│   │   ├── bm25.py                # BM25Manager (lexical search)
│   │   ├── embedding_client.py    # EURI embeddings API client
│   │   ├── pinecone_manager.py    # Vector DB operations
│   │   ├── hybrid_retriever.py    # Hybrid search + reranking
│   │   └── reranker.py            # Cross-encoder reranking
│   │
│   ├── generator/                 # LLM generation
│   │   └── gpt_client.py          # EURI chat completion with smart prompting
│   │
│   ├── cache/                     # Caching
│   │   └── semantic_cache.py      # Semantic similarity caching
│   │
│   ├── memory/                    # Chat history management
│   │   ├── chat_memory.py         # Message storage and retrieval
│   │   └── session_manager.py     # Session management and summarization
│   │
│   ├── ingestion/                 # Document processing
│   │   └── loader.py              # PDF/DOCX/TXT loader with OCR
│   │
│   ├── preprocessing/             # Text processing
│   │   └── chunker.py             # Document chunking and cleaning
│   │
│   └── logger.py                  # Logging configuration
│
├── clinical-rag-frontend/         # Frontend (React)
│   ├── public/
│   │   └── index.html             # HTML template with initial spinner
│   ├── src/
│   │   ├── App.js                 # Main React component
│   │   ├── App.css                # Styles and animations
│   │   ├── index.js               # React entry point
│   │   └── index.css              # Global styles
│   ├── package.json               # Node dependencies
│   └── .gitignore                 # Git ignore (includes node_modules)
│
├── data/                          # Source documents
│   └── *.pdf, *.docx, *.txt       # Clinical documents for indexing
│
├── tests/                         # Evaluation
│   ├── evaluate_metrics.py        # Metrics computation
│   └── evaluation_dataset.json    # Test questions and answers
│
├── bm25_index.pkl                 # Serialized BM25 index
├── requirements.txt               # Python dependencies
├── Dockerfile                     # API Docker image
├── .env                           # Environment variables (not committed)
├── .gitignore                     # Git ignore rules
├── main.py                        # Document indexing orchestration
└── README.md                      # This file
```

---

##  Quick Start

### Prerequisites

- **Python** 3.10+
- **Node.js** 18+
- **PostgreSQL** (or SQLite for dev)
- **API Keys**: EURI API, Pinecone

### 1. Backend Setup

```bash
# Clone repository
git clone <your-repo-url>
cd clinical-rag

# Create virtual environment
conda create -n rag python=3.10 -y
conda activate rag

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
# EURI / LLM
OPENAI_API_KEY=your_euri_api_key
EURI_EMBED_URI=https://api.euron.one/api/v1/euri/embeddings
EURI_CHAT_URI=https://api.euron.one/api/v1/euri/chat/completions

# Pinecone
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=clinical-rag
PINECONE_ENV=us-east-1

# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost/clinical_rag
# Or for SQLite: sqlite+aiosqlite:///./clinical_rag.db

# JWT Authentication
JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=10080
EOF

# Initialize database
python -c "from app.db.database import init_db; import asyncio; asyncio.run(init_db())"

# Index documents
python main.py

# Start API
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend Setup

```bash
# Navigate to frontend
cd clinical-rag-frontend

# Install dependencies
npm install

# Start development server
npm start
```

Frontend runs on `http://localhost:3000`

---

##  Environment Variables

### Backend (.env)

```bash
# ---- EURI / LLM ----
OPENAI_API_KEY=your_euri_api_key_here
EURI_EMBED_URI=https://api.euron.one/api/v1/euri/embeddings
EURI_CHAT_URI=https://api.euron.one/api/v1/euri/chat/completions

# ---- Pinecone ----
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=clinical-rag
PINECONE_ENV=us-east-1

# ---- Database ----
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/dbname
# Or: sqlite+aiosqlite:///./clinical_rag.db

# ---- JWT Authentication ----
JWT_SECRET_KEY=your-secret-key-from-secrets.token_urlsafe
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=10080  # 7 days

# ---- MLflow (optional) ----
MLFLOW_TRACKING_URI=azureml://...
AZUREML_EXPERIMENT_NAME=Clinical-RAG
```

### Frontend (.env.local - optional)

```bash
REACT_APP_API_URL=http://localhost:8000
```

---

##  Usage Guide

### First Time Setup

1. **Register Account**
   - Click "Register" tab
   - Enter email, username, password
   - Submit

2. **Login**
   - Enter email and password
   - Token stored in localStorage
   - Persistent across page reloads

3. **Ask Questions**
   - Type clinical/medical question
   - Press Enter or click Send
   - Watch real-time streaming response

### Features

**Bullet Points:**
```
"List the main health concerns in India as points"
```

**Tables:**
```
"Show me health worker density in India in table format"
```

**Chat History:**
- All conversations saved per user
- Click any session in sidebar to resume
- Smart titles auto-generated

---

## Authentication Flow

### Registration
```
POST /auth/register
{
  "email": "user@example.com",
  "username": "john_doe",
  "password": "SecurePass123",
  "full_name": "John Doe"
}
→ User created, password hashed with bcrypt
```

### Login
```
POST /auth/login
{
  "email": "user@example.com",
  "password": "SecurePass123"
}
→ Returns JWT token (expires in 7 days)
→ Frontend stores in localStorage
```

### Protected Requests
```
GET /sessions
Headers: { "Authorization": "Bearer eyJhbGc..." }
→ Backend verifies token
→ Returns user's sessions only
```

---

##  Frontend Features

### Modern UI Components

- **Authentication Screen**: Login/Register with loading states
- **Chat Interface**: Clean, ChatGPT-style layout
- **Sidebar**: Session list with smart titles
- **User Profile**: Avatar, name, logout
- **Message Bubbles**: User (blue) vs Assistant (gray)
- **Loading States**: 
  - Login: Spinning button
  - Sending: "Sending..." indicator
  - Typing: Three bouncing dots (● ● ●)
- **Tables**: HTML tables with hover effects
- **Bullet Points**: Automatic detection and styling

### Animations

- Fade-in messages
- Smooth transitions
- Loading spinners
- Pulsing text
- Hover effects

---

##  API Endpoints

### Authentication

```
POST   /auth/register       # Create new user
POST   /auth/login          # Login and get JWT token
GET    /auth/me             # Get current user info
```

### Chat Operations

```
POST   /ask                 # Ask question, stream response
GET    /sessions            # List user's chat sessions
GET    /sessions/{id}/messages   # Get session messages
```

### Request/Response Examples

**Ask Question:**
```json
// Request
POST /ask
{
  "question": "What are the components of Indian healthcare?",
  "session_id": "uuid-here"
}

// Response (streaming)
"The Indian healthcare system comprises..."
```

**List Sessions:**
```json
// Request
GET /sessions
Headers: { "Authorization": "Bearer ..." }

// Response
[
  {
    "id": "uuid-1",
    "title": "Indian Healthcare Components",
    "created_at": "2026-01-14T10:30:00"
  },
  ...
]
```

---

##  How It Works

### Document Indexing (One-time)

```
1. Load Documents (PDF/DOCX/TXT)
   ↓
2. OCR on images if needed
   ↓
3. Clean and chunk text (300 tokens, 120 overlap)
   ↓
4. Generate embeddings (EURI API)
   ↓
5. Store in Pinecone (vector search)
   ↓
6. Build BM25 index (keyword search)
   ↓
7. Save BM25 to bm25_index.pkl
```

### Question Answering (Runtime)

```
1. User asks question
   ↓
2. Check semantic cache (instant if cached)
   ↓
3. Hybrid retrieval:
   - BM25 search (top 50)
   - Vector search (top 50)
   - Merge with RRF
   ↓
4. Cross-encoder reranking (top 5)
   ↓
5. Build context with chat history summary
   ↓
6. LLM generation (streaming)
   ↓
7. Save to database
   ↓
8. Cache response
   ↓
9. Stream to frontend
```

---

## Evaluation

### Metrics Computed

- **Recall**: Percentage of relevant docs retrieved
- **Precision**: Percentage of retrieved docs that are relevant
- **Hallucination Rate**: Answers not grounded in context
- **Faithfulness Score**: Cosine similarity between answer and context

### Run Evaluation

```bash
python tests/evaluate_metrics.py
```

### Results Logged to MLflow

Configure MLflow tracking URI in `.env` to track experiments.

---

## Docker Deployment

### Build Images

**Backend:**
```bash
docker build -t clinical-rag-api .
docker tag clinical-rag-api <registry>/clinical-rag-api:v1
docker push <registry>/clinical-rag-api:v1
```

**Frontend:**
```bash
cd clinical-rag-frontend
docker build -t clinical-rag-frontend .
docker tag clinical-rag-frontend <registry>/clinical-rag-frontend:v1
docker push <registry>/clinical-rag-frontend:v1
```

### Run Locally

```bash
# Backend
docker run -p 8000:8000 --env-file .env clinical-rag-api

# Frontend
docker run -p 3000:3000 -e REACT_APP_API_URL=http://localhost:8000 clinical-rag-frontend
```

---

##  Azure Deployment

### Backend (Azure Web App)

```bash
# Deploy to Azure Web App for Containers
az webapp create \
  --resource-group <rg> \
  --plan <plan> \
  --name clinical-rag-api \
  --deployment-container-image-name <acr>/clinical-rag-api:v1

# Configure app settings
az webapp config appsettings set \
  --name clinical-rag-api \
  --settings \
    WEBSITES_PORT=8000 \
    JWT_SECRET_KEY="..." \
    DATABASE_URL="..." \
    # ... all other env vars
```

### Frontend (Vercel/Netlify)

**Vercel:**
```bash
cd clinical-rag-frontend
vercel
# Add REACT_APP_API_URL in dashboard
```

**Netlify:**
```bash
cd clinical-rag-frontend
npm run build
netlify deploy --prod --dir=build
```

---

## 🔧 Development

### Backend Development

```bash
# Auto-reload on code changes
uvicorn app.main:app --reload --port 8000

# Check API docs
open http://localhost:8000/docs
```

### Frontend Development

```bash
npm start  # Auto-reload on changes
```

### Database Migrations

```bash
# Using Alembic (recommended)
alembic revision --autogenerate -m "Add new table"
alembic upgrade head

# Or manual SQL
psql -d clinical_rag -f migrations/001_add_users.sql
```

---

##  Testing

### Manual Testing

- [ ] Register new user
- [ ] Login with credentials
- [ ] Ask a question
- [ ] See streaming response
- [ ] Check table rendering
- [ ] Verify bullet points
- [ ] Resume previous chat
- [ ] Logout and login again

### API Testing

```bash
# Register
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","username":"test","password":"test123"}'

# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"test123"}'
```

---

##  Performance Optimizations

### Implemented

- ✅ Semantic caching (instant for repeated questions)
- ✅ Database indexing on user_id, session_id
- ✅ Streaming responses (perceived performance)
- ✅ Hybrid retrieval (better accuracy than vector-only)
- ✅ Cross-encoder reranking (precision boost)
- ✅ Conversation summarization (context management)

### Potential Improvements

- [ ] Redis caching layer
- [ ] CDN for frontend assets
- [ ] Database connection pooling
- [ ] Async batch embedding
- [ ] Query result pagination

---

## Security Best Practices

### Implemented

- ✅ JWT authentication with secure tokens
- ✅ Bcrypt password hashing
- ✅ CORS configuration
- ✅ Environment variables for secrets
- ✅ User session isolation
- ✅ Token expiration (7 days)

### For Production

- [ ] HTTPS only (SSL/TLS)
- [ ] Rate limiting
- [ ] Input validation and sanitization
- [ ] SQL injection protection (using ORM)
- [ ] XSS protection
- [ ] CSRF tokens
- [ ] Security headers
- [ ] Audit logging

---

## Roadmap

### Completed 

- [x] End-to-end RAG pipeline
- [x] Hybrid retrieval (BM25 + Vector)
- [x] Cross-encoder reranking
- [x] JWT authentication
- [x] User management
- [x] Chat history per user
- [x] React frontend
- [x] Streaming responses
- [x] Smart formatting (tables, bullets)
- [x] Semantic caching
- [x] Conversation summarization

### In Progress 

- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Comprehensive test suite
- [ ] API documentation (Swagger enhancements)

### Planned 

- [ ] OAuth2 integration (Google, Microsoft)
- [ ] Role-based access control (RBAC)
- [ ] Multi-language support
- [ ] Voice input/output
- [ ] Export chat history (PDF, Markdown)
- [ ] Advanced analytics dashboard
- [ ] Document upload through UI
- [ ] Collaborative chat sessions
- [ ] Application Insights monitoring

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

##  License

This project is licensed under the MIT License - see LICENSE file for details.

---

##  Authors

- **Your Name** - Initial work and architecture
- **Contributors** - See contributor list

---

##  Acknowledgments

- **EURI API** - LLM and embeddings
- **Pinecone** - Vector database
- **FastAPI** - Modern Python web framework
- **React** - Frontend library
- **Sentence Transformers** - Embeddings and reranking
- **OpenAI** - Inspiration for chat interface

---

## Support

- **Documentation**: See `/docs` folder
- **Issues**: GitHub Issues
- **Email**: borasanthosh921@gmail.com

---

**Built with ❤️ for better healthcare information access**

Last Updated: January 2026