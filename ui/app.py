import streamlit as st
import requests
import os
import uuid

# ================= CONFIG =================
API_URL = os.getenv("API_URL", "http://localhost:8000")
ASK_URL = f"{API_URL}/ask"

st.set_page_config(page_title="Clinical RAG Assistant", layout="wide")

# ================= STYLES =================
st.markdown("""
<style>
.chat-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 8px;
}
.chat-user .bubble {
    background-color: #1f2937;
    padding: 10px 14px;
    border-radius: 12px;
    max-width: 70%;
}

.chat-assistant {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 8px;
}
.chat-assistant .bubble {
    background-color: #111827;
    padding: 10px 14px;
    border-radius: 12px;
    max-width: 70%;
}
</style>
""", unsafe_allow_html=True)

# ================= SESSION STATE =================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "sessions_cache" not in st.session_state:
    st.session_state.sessions_cache = []

# ================= HELPERS =================
def fetch_sessions_cached():
    """Fetch once, reuse on reruns"""
    try:
        resp = requests.get(f"{API_URL}/sessions", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                st.session_state.sessions_cache = data
    except Exception:
        pass
    return st.session_state.sessions_cache

def fetch_messages(session_id):
    try:
        resp = requests.get(f"{API_URL}/sessions/{session_id}/messages", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []

def ask_backend_stream(question: str):
    try:
        with requests.post(
            ASK_URL,
            json={"question": question, "session_id": st.session_state.session_id},
            stream=True,
            timeout=(5, 90)
        ) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=32, decode_unicode=True):
                if chunk:
                    yield chunk
    except Exception as e:
        yield f"\n⚠️ {e}"

# ================= MAIN UI =================
st.title("🩺 Clinical RAG Assistant")

# -------- Chat History --------
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div class="chat-user">
                <div class="bubble">{msg["content"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="chat-assistant">
                <div class="bubble">
            """,
            unsafe_allow_html=True
        )
        # ✅ Use markdown so tables render
        st.markdown(msg["content"])
        st.markdown("</div></div>", unsafe_allow_html=True)

# -------- Input Form (CORRECT WAY) --------
with st.form("chat_form", clear_on_submit=True):
    question = st.text_input("Ask a clinical / medical question")
    submitted = st.form_submit_button("Send")

if submitted and question.strip():
    # user message
    st.session_state.chat_history.append(
        {"role": "user", "content": question}
    )

    placeholder = st.empty()
    answer = ""

    for chunk in ask_backend_stream(question):
        answer += chunk
        placeholder.markdown(
            """
            <div class="chat-assistant">
                <div class="bubble">
            """,
            unsafe_allow_html=True
        )
        st.markdown(answer)
        st.markdown("</div></div>", unsafe_allow_html=True)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )

    st.rerun()

# ================= SIDEBAR =================
with st.sidebar:
    st.title("💬 Chats")

    if st.button("➕ New Chat"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    sessions = fetch_sessions_cached()

    for s in sessions:
        if not isinstance(s, dict):
            continue
        if st.button(s["title"], key=s["id"]):
            st.session_state.session_id = s["id"]
            st.session_state.chat_history = fetch_messages(s["id"])
            st.rerun()
