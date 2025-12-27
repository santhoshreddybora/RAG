from sqlalchemy.future import select
from app.db.models import ChatMessage, ChatSummary

LAST_N = 6  # configurable

async def save_message(db, session_id, role, content):
    msg = ChatMessage(
        session_id=session_id,
        role=role,
        content=content
    )
    db.add(msg)
    await db.commit()

async def get_last_n_messages(db, session_id):
    q = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(LAST_N)
    )
    res = await db.execute(q)
    messages = res.scalars().all()
    return [
        {"role":m.role,"content":m.content} for m in reversed(messages)
        ]
async def get_summary(db, session_id):
    q = select(ChatSummary).where(ChatSummary.session_id == session_id)
    res = await db.execute(q)
    return res.scalar_one_or_none()

async def save_summary(db, session_id, summary_text):
    existing = await get_summary(db, session_id)
    if existing:
        existing.summary = summary_text
    else:
        new_summary = ChatSummary(
            session_id=session_id,
            summary=summary_text
        )
        db.add(new_summary)
    await db.commit()


