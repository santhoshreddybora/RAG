from sqlalchemy import Column, String, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base
import uuid

Base = declarative_base()

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, server_default=func.now())

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id"))
    role = Column(String)  # user / assistant
    content = Column(Text)
    created_at = Column(DateTime, server_default=func.now())

class ChatSummary(Base):
    __tablename__ = "chat_summaries"
    session_id = Column(UUID(as_uuid=True), primary_key=True)
    summary = Column(Text)
