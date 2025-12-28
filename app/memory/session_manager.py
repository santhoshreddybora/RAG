from sqlalchemy.future import select
from sqlalchemy import func
from app.db.models import ChatSession,ChatMessage
import uuid
from app.db.models import ChatSession
from uuid import UUID

async def ensure_session(db, session_id: UUID):
    result = await db.execute(
        select(ChatSession.id).where(ChatSession.id == session_id)
    )
    session = result.scalar_one_or_none()

    if session is None:
        session = ChatSession(id=session_id)
        db.add(session)
        await db.commit()

    return session

async def is_first_message(db, session_id: UUID) -> bool:
    """
    Returns True if no messages exist yet for this session
    """
    result = await db.execute(
        select(func.count(ChatMessage.id))
        .where(ChatMessage.session_id == session_id)
        .limit(1)
    )
    count=result.scalar()
    return count==0


async def set_session_title(db, session_id: UUID, title: str):
    session = await db.get(ChatSession, session_id)
    if not session:
        return
    session.title = title
    await db.commit()
