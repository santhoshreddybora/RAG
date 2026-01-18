from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.retrieval.hybrid_retriever import HybridRetriever
from app.generator.gpt_client import GPTClient
from app.cache.semantic_cache import SemanticCache
from fastapi.responses import StreamingResponse
import time
from app.memory.chat_memory import (
    get_last_n_messages,
    save_message_bulk,
    get_summary
)
from sqlalchemy.ext.asyncio import AsyncSession
from app.memory.session_manager import update_and_save_summary
from app.db.database import get_db,AsyncSessionLocal
from app.memory.session_manager import ensure_session_and_check_first, set_session_title
from app.db.models import ChatSession, ChatMessage, User
from sqlalchemy.future import select
from uuid import UUID
from app.logger import logging
from app.retrieval.embedding_client import EuriEmbeddingClient
from app.auth.auth_utils import get_current_active_user  # NEW
from app.routers import auth  # NEW
import asyncio
from contextlib import asynccontextmanager
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

semantic_cache = SemanticCache()

app = FastAPI(title="Clinical RAG API")
@app.on_event("startup")
async def startup_event():
    """Configure connection pooling for better performance"""
    
    # Create a shared session with connection pooling
    session = requests.Session()
    
    # Retry strategy (only for connection errors, not timeouts)
    retry_strategy = Retry(
        total=0,  # We handle retries manually
        connect=0,
        read=0,
        status_forcelist=[],
    )
    
    adapter = HTTPAdapter(
        pool_connections=20,
        pool_maxsize=40,
        max_retries=retry_strategy
    )
    
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    logging.info("✅ Connection pooling configured")
# Include auth routes
app.include_router(auth.router)  # NEW

retriever = HybridRetriever()
llm = GPTClient()
emb_client = EuriEmbeddingClient()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    history: list = []
    session_id: str = None

@asynccontextmanager
async def log_request_time(step_name: str, start_time: float):
    """Log time taken for a specific step"""
    step_start = time.time()
    try:
        yield
    finally:
        step_elapsed = time.time() - step_start
        total_elapsed = time.time() - start_time
        logging.info(f"⏱️  {step_name}: {step_elapsed:.2f}s | Total: {total_elapsed:.2f}s")

async def background_save_messages(session_id: UUID, messages: list):
    """Save messages in a new database session"""
    async with AsyncSessionLocal() as db:
        try:
            await save_message_bulk(db, session_id, messages)
            await db.commit()  # Important: commit the transaction
        except Exception as e:
            logging.error(f"Failed to save messages in background: {e}")
            await db.rollback()

async def background_update_summary(session_id: UUID, existing_summary: str, messages: list):
    """Update summary in a new database session"""
    try:
        await update_and_save_summary(session_id, existing_summary, messages)
    except Exception as e:
        logging.error(f"Failed to update summary in background: {e}")

def stream_cached_answer(answer: str):
    for word in answer.split():
        yield word + " "

def chunk_text(text: str, chunk_size=20):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size]) + " "
        time.sleep(0.05)

@app.post("/ask")
async def ask_question(
    req: QueryRequest,
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)  # NEW - Require auth
):
    t0 = time.time()
    
    # Step 1: Generate embedding
    async with log_request_time("1. Embedding generation", t0):
        query_embedding = emb_client.embed([req.question])
        if not query_embedding:
            raise HTTPException(status_code=503, detail="Embedding failed")
        query_embedding = query_embedding[0]
    
    # Step 2: Session check/creation
    async with log_request_time("2. Session check/creation", t0):
        session_id = UUID(req.session_id)
        
        result = await db.execute(
            select(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.user_id == current_user.id
            )
        )
        session = result.scalar_one_or_none()
        
        if not session:
            session = ChatSession(
                id=session_id,
                user_id=current_user.id,
                title=req.question[:50]
            )
            db.add(session)
            await db.commit()
            is_new = True
        else:
            is_new = False
        
        if is_new:
            try:
                title = llm.generate_title(req.question)
            except:
                title = req.question[:50]
            await set_session_title(db, session_id, title)
    
    # Step 3: Check cache
    async with log_request_time("3. Cache check", t0):
        cached_answer = semantic_cache.get(req.session_id, req.question, query_embedding)
        if cached_answer:
            logging.info("✅ Cache HIT - returning cached answer")
            return StreamingResponse(
                stream_cached_answer(cached_answer),
                media_type="text/plain"
            )
        logging.info("❌ Cache MISS - proceeding to generate")
    
    # Step 4: Fetch messages and summary
    async with log_request_time("4. Fetch messages & summary", t0):
        last_messages_task = asyncio.create_task(get_last_n_messages(db, session_id))
        summary_obj_task = asyncio.create_task(get_summary(db, session_id))
        last_messages, summary_obj = await asyncio.gather(
            last_messages_task, summary_obj_task
        )
        summary_text = summary_obj.summary if summary_obj else None
    
    # Step 5: Hybrid retrieval
    async with log_request_time("5. Hybrid retrieval (BM25 + Vector + Rerank)", t0):
        contexts = await retriever.hybrid_search(req.question, query_embedding)
        logging.info(f"📄 Retrieved {len(contexts)} context chunks")
    
    # Step 6: Build history
    async with log_request_time("6. Build conversation history", t0):
        history = []
        if summary_text:
            history.append({"role": "system", "content": summary_text})
        
        history.extend([
            {"role": m['role'] if isinstance(m, dict) else m.role,
             "content": m['content'] if isinstance(m, dict) else m.content}
            for m in last_messages
        ])
    
    # Step 7: LLM generation
    async with log_request_time("7. LLM generation", t0):
        answer = llm.generate_text(req.question, contexts, history=history)
        
        if answer.startswith("Sorry"):
            logging.warning("⚠️  LLM returned error message")
            return StreamingResponse(iter([answer]), media_type="text/plain")
    
    # Step 8: Save to database (background)
    async with log_request_time("8. Schedule background tasks", t0):
        background_tasks.add_task(
            background_save_messages,
            session_id,
            [('user', req.question), ('assistant', answer)]
        )
        
        if len(last_messages) >= 6:
            background_tasks.add_task(
                background_update_summary,
                session_id,
                existing_summary=summary_text,
                messages=last_messages
            )
        
        # Cache the answer
        if (isinstance(answer, str) and 'Error' not in answer and 'No contexts' not in answer):
            semantic_cache.set(req.session_id, req.question, answer, query_embedding)
    
    # Final log
    total_time = time.time() - t0
    logging.info(f"🎉 REQUEST COMPLETE - Total time: {total_time:.2f}s")
    
    return StreamingResponse(chunk_text(answer), media_type="text/plain")
    

@app.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)  # NEW - Require auth
):
    # Verify session belongs to user
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id  # NEW - Security check
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
    )
    messages = result.scalars().all()
    
    return [
        {"role": m.role, "content": m.content}
        for m in messages
    ]

@app.get("/sessions")
async def list_sessions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)  # NEW - Require auth
):
    try:
        result = await db.execute(
            select(ChatSession)
            .where(ChatSession.user_id == current_user.id)  # NEW - Only user's sessions
            .order_by(ChatSession.created_at.desc())
        )
        sessions = result.scalars().all()
        return [
            {"id": str(s.id), "title": s.title, "created_at": s.created_at}
            for s in sessions
        ]
    except Exception as e:
        logging.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")