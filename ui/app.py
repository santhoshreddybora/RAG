import streamlit as st
import requests
import os
from datetime import datetime

# =============================
# 🔧 CONFIG
# =============================
API_URL = os.getenv("API_URL", "http://localhost:8000")
ASK_URL = f"{API_URL}/ask"
FEEDBACK_URL = f"{API_URL}/feedback"

# =============================
#  SESSION STATE INIT
# =============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =============================
#  PAGE SETUP
# =============================
st.set_page_config(page_title="Clinical RAG Assistant", layout="wide")
st.title("🩺 Clinical RAG Assistant")

# =============================
#  DISPLAY CHAT HISTORY
# =============================
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

# =============================
# USER INPUT
# =============================
question = st.text_input("Ask a clinical/medical question")

# =============================
# BACKEND CALL
# =============================
def ask_backend(question: str):
    res = requests.post(ASK_URL, json={"question": question})
    if res.status_code != 200:
        return None, None
    data = res.json()
    return data["answer"], data["contexts"]

# =============================
# SUBMIT BUTTON
# =============================
if st.button("Send"):
    if question.strip():
        # 1️⃣ Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.spinner("🤖 Thinking..."):
            answer, contexts = ask_backend(question)

        # 2️⃣ Add assistant message
        if answer:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })

        # 3️⃣ Rerun to update UI
        st.rerun()

# =============================
# CLEAR CHAT
# =============================
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()
