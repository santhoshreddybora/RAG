import streamlit as st
import requests
import os
# =============================
# 🔧 CONFIG — FastAPI URL
# =============================
API_URL = os.getenv("API_URL")
ASK_URL = f"{API_URL}/ask"
FEEDBACK_URL = f"{API_URL}/feedback"

# =============================
# 🎨 STREAMLIT PAGE
# =============================
st.set_page_config(page_title="Clinical RAG Assistant", layout="wide")
st.title("🩺 Clinical RAG Assistant 🤖")

# Input box
question = st.text_input("🔍 Ask a clinical/medical question")


# =============================
# 🚀 CALL FASTAPI /ask
# =============================
def ask_backend(question: str):
    try:
        res = requests.post(ASK_URL, json={"question": question})
        if res.status_code != 200:
            st.error("❌ Backend Error!")
            return None, None
        data = res.json()
        return data["answer"], data["contexts"]
    except Exception as e:
        st.error(f"⚠️ Request failed: {e}")
        return None, None


# =============================
# 🎭 Show Result + Sources
# =============================
if st.button("Get Answer"):
    if not question.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("🤖 Thinking..."):
            answer, contexts = ask_backend(question)

        if answer:
            st.subheader("🧠 Answer")
            st.write(answer)

            st.subheader("📚 Sources")
            for i, ctx in enumerate(contexts or [], 1):
                st.markdown(f"**Source {i}:** {ctx[:300]}...")

            # Feedback
            st.subheader("📝 Was this answer helpful?")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("👍 Yes"):
                    requests.post(FEEDBACK_URL, json={
                        "question": question, "answer": answer, "helpful": True
                    })
                    st.success("Thank you for the feedback! 💙")

            with col2:
                if st.button("👎 No"):
                    requests.post(FEEDBACK_URL, json={
                        "question": question, "answer": answer, "helpful": False
                    })
                    st.warning("Feedback noted! 👀")
