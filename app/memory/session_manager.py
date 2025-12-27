from sqlalchemy.future import select
from app.db.models import ChatSession
import uuid

async def ensure_session(db, session_id: str):
    result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id)
    )
    session = result.scalar_one_or_none()

    if session is None:
        session = ChatSession(id=session_id)
        db.add(session)
        await db.commit()

    return session
