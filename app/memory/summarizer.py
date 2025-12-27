# app/memory/summarizer.py
from typing import List


async def update_summary(llm, existing_summary: str, messages: List[dict]) -> str:
    """
    Compress conversation into a short summary
    """

    conversation_text = ""
    for m in messages:
        conversation_text += f"{m['role']}: {m['content']}\n"

    prompt = f"""
    You are summarizing a conversation for memory compression.

    Existing summary:
    {existing_summary or "None"}

    New conversation messages:
    {conversation_text}

    Update the summary in 4–6 sentences.
    Focus on:
    - User intent
    - Important facts
    - Decisions made
    - Preferences
    """

    summary = llm.summarize(prompt)
    return summary
