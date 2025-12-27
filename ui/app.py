import streamlit as st
import requests
import os
from datetime import datetime

#  CONFIG
API_URL = os.getenv("API_URL", "http://localhost:8000")
ASK_URL = f"{API_URL}/ask"
FEEDBACK_URL = f"{API_URL}/feedback"
import uuid

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

#  SESSION STATE INIT
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

#  PAGE SETUP
st.set_page_config(page_title="Clinical RAG Assistant", layout="wide")
st.title("🩺 Clinical RAG Assistant")

#  DISPLAY CHAT HISTORY
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div class="chat-message user">
                <b>You:</b><br>{msg["content"]}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="chat-message assistant">
                <b>Assistant:</b><br>{msg["content"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

# USER INPUT
question = st.text_input("Ask a clinical/medical question")

# BACKEND CALL
def ask_backend(question: str):
    res =requests.post(
            ASK_URL,
            json={
                "question": question,
                "session_id": st.session_state.session_id
                }
            )

    if res.status_code != 200:
        return None, None
    data = res.json()
    return data["answer"], data["contexts"]

def ask_backend_stream(question: str):
    with requests.post(
        ASK_URL,
        json={
            "question": question,
            "session_id": st.session_state.session_id
        },
        stream=True,
        timeout=(5, 60)
    ) as r:
        for chunk in r.iter_content(chunk_size=32, decode_unicode=True):
            if chunk:
                yield chunk

# SUBMIT BUTTON
if st.button("Send"):
    if question.strip():
        # 1️⃣ Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        assisant_placeholder=st.empty()
        streamed_answer=""

        for chunk in ask_backend_stream(question):
            streamed_answer +=chunk
            assisant_placeholder.markdown(
                f"""
                <div class="chat-message assistant">
                    <b>Assistancct:</b><br>{streamed_answer}
                </div>""",
                unsafe_allow_html=True
            )

        # 2️⃣ Add assistant message
        st.session_state.chat_history.append({
                "role": "assistant",
                "content": streamed_answer
            })
        # 3️⃣ Rerun to update UI
        st.rerun()

if st.button("➕ New Chat"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.rerun()

# CLEAR CHAT
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()
