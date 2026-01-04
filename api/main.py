from fastapi import FastAPI,HTTPException,BackgroundTasks
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
from fastapi import Depends
from app.memory.session_manager import update_and_save_summary
from app.db.database import get_db
from app.memory.session_manager import ensure_session_and_check_first,set_session_title
from app.db.models import ChatSession,ChatMessage
from sqlalchemy.future import select
from uuid import UUID
from app.logger import logging
from app.retrieval.embedding_client import EuriEmbeddingClient
import asyncio
semantic_cache=SemanticCache()

app = FastAPI(title="Clinical RAG API")
retriever = HybridRetriever()
llm = GPTClient()
emb_client=EuriEmbeddingClient()

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
async def ask_question(req: QueryRequest,background_tasks:BackgroundTasks=None,db:AsyncSession=Depends(get_db)):
    # session_id=req.session_id
    t0=time.time()
    query_embedding = emb_client.embed([req.question])
    if not query_embedding:
        raise HTTPException(status_code=503, detail="Embedding failed")
    query_embedding = query_embedding[0]
    logging.info(f"computed query embedding {time.time()-t0}")
    session_id = UUID(req.session_id)
    logging.info(f"started time for session {time.time()-t0}")
    session,is_new = await ensure_session_and_check_first(db,session_id)
    logging.info(f"ensured session and first message exists {time.time()-t0}")

    if is_new:
        title = req.question[:50]  # trim
        await set_session_title(db, session_id, title)
    logging.info(f"set session title if new {time.time()-t0}")

    # ## Check Cache first 
    cached_answer = semantic_cache.get(req.session_id, req.question,query_embedding)
    logging.info(f"checked cache {time.time()-t0}")
    if cached_answer:
        return StreamingResponse(
            stream_cached_answer(cached_answer),
            media_type="text/plain"
        )
    ## Get the last messages and summary from DB
    logging.info(f"no cache hit, proceeding to generate answer {time.time()-t0}")
    last_messages_task = asyncio.create_task(get_last_n_messages(db,session_id))
    summary_obj_task = asyncio.create_task(get_summary(db,session_id))
    last_messages,summary_obj=await asyncio.gather(
        last_messages_task,summary_obj_task
    )
    logging.info(f"summary_obj: {summary_obj.summary} and last_messages are {last_messages}")
    summary_text=summary_obj.summary if summary_obj else None
    logging.info(f"fetched last messages and summary {time.time()-t0}")

    ##get the RAG contexts 
    contexts = await retriever.hybrid_search(req.question,query_embedding)
    logging.info(f"retrieved contexts {time.time()-t0}")
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
    if answer.startswith("Sorry"):
        return StreamingResponse(iter([answer]), media_type="text/plain")
    
    logging.info(f"generated LLM answer {time.time()-t0}")
    ##save messages and answer to DB
    # await save_message(db,session_id,"user",req.question)
    # await save_message(db,session_id,"assistant",answer)
    background_tasks.add_task(
        save_message_bulk,
        db,
        session_id,
        [
            ('user',req.question),
            ('assistant',answer)
        ]
    )
    logging.info(f"saved messages to DB {time.time()-t0}")

    ##update summary 
    if len(last_messages) >= 6:
        background_tasks.add_task(update_and_save_summary,
            session_id,
            existing_summary=summary_text,
            messages=last_messages
        )

        logging.info(f"updated and saved summary {time.time()-t0}")
    
    if (isinstance(answer, str) and 'Error' not in answer and  'No contexts' not in answer):
        semantic_cache.set(req.session_id,req.question,answer,query_embedding)

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
    try:
        result = await db.execute(
            select(ChatSession)
            .order_by(ChatSession.created_at.desc())
        )
        sessions = result.scalars().all()
        return [
            {"id": s.id, "title": s.title, "created_at": s.created_at}
            for s in sessions
        ]
    except Exception as e:
        logging.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")








# class Feedback(BaseModel):
#     question: str
#     answer: str
#     helpful: bool

# @app.post("/feedback")
# def save_feedback(data: Feedback):
#     with open("feedback.log", "a") as f:
#         f.write(f"{data.question}|{data.answer}|{data.helpful}\n")
#     return {"status": "saved"}
