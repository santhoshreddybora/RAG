from sqlalchemy.future import select
from sqlalchemy import func
from app.db.models import ChatSession,ChatMessage
import uuid
from app.db.models import ChatSession
from uuid import UUID
from app.db.database import AsyncSessionLocal
from app.memory.chat_memory import save_summary,get_summary
from app.memory.summarizer import update_summary
from app.generator.gpt_client import GPTClient
from app.logger import logging

async def update_and_save_summary(session_id, existing_summary, messages):
    async with AsyncSessionLocal() as db:
        try:
            # Create fresh LLM instance for this task
            llm = GPTClient()  # ✅ GOOD: Fresh instance
            
            new_summary = await update_summary(
                llm=llm,
                existing_summary=existing_summary,
                messages=messages
            )
            await save_summary(db, session_id, new_summary)
            await db.commit()  # Ensure commit
            
        except Exception as e:
            logging.error(f"Failed to update summary: {e}")
            await db.rollback()

async def ensure_session_and_check_first(db, session_id):
    # Check if session exists
    session = await db.get(ChatSession, session_id)

    if not session:
        session = ChatSession(id=session_id)
        db.add(session)
        await db.commit()
        return session, True  # first message

    # Check if messages exist
    result = await db.execute(
        select(func.count(ChatMessage.id))
        .where(ChatMessage.session_id == session_id)
    )
    count = result.scalar()
    return session, count == 0

async def set_session_title(db, session_id: UUID, title: str):
    session = await db.get(ChatSession, session_id)
    if not session:
        return
    session.title = title
    await db.commit()



