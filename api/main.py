from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.retrieval.hybrid_retriever import HybridRetriever
from app.generator.gpt_client import GPTClient
from app.cache.semantic_cache import SemanticCache
from fastapi.responses import StreamingResponse
import time
from app.memory.chat_memory import (
    get_last_n_messages,
    save_message,
    get_summary,
    save_summary
)
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends
from app.memory.summarizer import update_summary
from app.db.database import get_db
from app.memory.session_manager import ensure_session,is_first_message,set_session_title
from app.db.models import ChatSession,ChatMessage
from sqlalchemy.future import select
from uuid import UUID


semantic_cache=SemanticCache()

app = FastAPI(title="Clinical RAG API")
retriever = HybridRetriever()
llm = GPTClient()

# # Allow React local dev (3000)
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
    session_id:str=None


def stream_cached_answer(answer: str):
    for word in answer.split():
        yield word + " "

def chunk_text(text: str, chunk_size=20):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size]) + " "
        time.sleep(0.05) 


@app.post("/ask")
async def ask_question(req: QueryRequest,db:AsyncSession=Depends(get_db)):
    # session_id=req.session_id
    session_id = UUID(req.session_id)
    await ensure_session(db, session_id)
    is_new = await is_first_message(db, session_id)

    if is_new:
        title = req.question[:50]  # trim
        await set_session_title(db, session_id, title)

    # ## Check Cache first 
    # cache_key = f"{session_id}:{req.question}"
    cached_answer = semantic_cache.get(req.session_id, req.question)
    if cached_answer:
        return StreamingResponse(
            stream_cached_answer(cached_answer),
            media_type="text/plain"
        )
    ## Get the last messages and summary from DB
    last_messages = await get_last_n_messages(db,session_id)
    summary_obj = await get_summary(db,session_id)
    summary_text=summary_obj.summary if summary_obj else None


    ##get the RAG contexts 
    contexts = await retriever.hybrid_search(req.question)

    ## LLM answer 

    history = []

    if summary_text:
        history.append({
            "role": "system",
            "content": summary_text
        })

    history.extend([
        {"role": m['role']if isinstance(m,dict) else  m.role, 
         "content": m['content'] if isinstance(m,dict) else m.content}
        for m in last_messages
    ])


    answer = llm.generate_text(req.question, contexts,
                               history=history)
    
    ##save messages and answer to DB
    await save_message(db,session_id,"user",req.question)
    await save_message(db,session_id,"assistant",answer)


    ##update summary 
    if len(last_messages) >= 6:
        new_summary = await update_summary(
            llm=llm,
            existing_summary=summary_text,
            messages=last_messages
        )
        await save_summary(db,session_id, new_summary)

    if answer and isinstance(answer, str) and not answer.lower().startswith("error"):
        semantic_cache.set(req.session_id,req.question,answer)

    # return {"answer": answer, "contexts": contexts,"cached":False}
    return StreamingResponse(
        chunk_text(answer),
        media_type="text/plain"
    )

@app.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: UUID,
    db: AsyncSession = Depends(get_db)
):
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
async def list_sessions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ChatSession)
        .order_by(ChatSession.created_at.desc())
    )
    sessions = result.scalars().all()
    return [
        {"id": s.id, "title": s.title, "created_at": s.created_at}
        for s in sessions
    ]








# class Feedback(BaseModel):
#     question: str
#     answer: str
#     helpful: bool

# @app.post("/feedback")
# def save_feedback(data: Feedback):
#     with open("feedback.log", "a") as f:
#         f.write(f"{data.question}|{data.answer}|{data.helpful}\n")
#     return {"status": "saved"}
