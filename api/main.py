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
from app.db.database import get_db
from app.memory.session_manager import ensure_session_and_check_first, set_session_title
from app.db.models import ChatSession, ChatMessage, User
from sqlalchemy.future import select
from uuid import UUID
from app.logger import logging
from app.retrieval.embedding_client import EuriEmbeddingClient
from app.auth.auth_utils import get_current_active_user  # NEW
from app.routers import auth  # NEW
import asyncio

semantic_cache = SemanticCache()

app = FastAPI(title="Clinical RAG API")

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
    query_embedding = emb_client.embed([req.question])
    logging.info(f"time taken for embedding: {time.time() - t0} seconds")
    if not query_embedding:
        raise HTTPException(status_code=503, detail="Embedding failed")
    query_embedding = query_embedding[0]
    
    session_id = UUID(req.session_id)
    
    # Verify session belongs to current user
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id  # NEW - Security check
        )
    )

    session = result.scalar_one_or_none()
    
    if not session:
        # Create new session for this user
        session = ChatSession(
            id=session_id,
            user_id=current_user.id,  # NEW
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
    logging.info(f"Time taken for session check/creation: {time.time() - t0} seconds")
    # Check Cache
    cached_answer = semantic_cache.get(req.session_id, req.question, query_embedding)
    if cached_answer:
        return StreamingResponse(
            stream_cached_answer(cached_answer),
            media_type="text/plain"
        )
    logging.info(f"Cache miss, proceeding to generate answer. Time elapsed: {time.time() - t0} seconds")
    # Get messages and summary
    last_messages_task = asyncio.create_task(get_last_n_messages(db, session_id))
    summary_obj_task = asyncio.create_task(get_summary(db, session_id))
    last_messages, summary_obj = await asyncio.gather(
        last_messages_task, summary_obj_task
    )
    logging.info(f"Time taken to fetch messages and summary: {time.time() - t0} seconds")
    summary_text = summary_obj.summary if summary_obj else None
    
    # Get RAG contexts
    contexts = await retriever.hybrid_search(req.question, query_embedding)
    logging.info(f"Time taken to retrieve contexts: {time.time() - t0} seconds")
    # Build history
    history = []
    if summary_text:
        history.append({"role": "system", "content": summary_text})
    
    history.extend([
        {"role": m['role'] if isinstance(m, dict) else m.role,
         "content": m['content'] if isinstance(m, dict) else m.content}
        for m in last_messages
    ])
    
    # Generate answer
    answer = llm.generate_text(req.question, contexts, history=history)
    logging.info(f"Time taken to generate answer: {time.time() - t0} seconds")
    if answer.startswith("Sorry"):
        return StreamingResponse(iter([answer]), media_type="text/plain")
    
    # Save messages
    background_tasks.add_task(
        save_message_bulk,
        db,
        session_id,
        [('user', req.question), ('assistant', answer)]
    )
    
    # Update summary if needed
    if len(last_messages) >= 6:
        background_tasks.add_task(
            update_and_save_summary,
            session_id,
            existing_summary=summary_text,
            messages=last_messages
        )
    logging.info(f"Time taken to schedule background tasks for save message and update summary: {time.time() - t0} seconds")
    if (isinstance(answer, str) and 'Error' not in answer and 'No contexts' not in answer):
        semantic_cache.set(req.session_id, req.question, answer, query_embedding)
    
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