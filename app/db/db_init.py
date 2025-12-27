import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.db.database import engine
from app.db.models import Base

async def init():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(init())
