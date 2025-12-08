from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.retrieval.hybrid_retriever import HybridRetriever
from app.generator.gpt_client import GPTClient

app = FastAPI(title="Clinical RAG API")


# # Allow React local dev (3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = HybridRetriever()
llm = GPTClient()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(req: QueryRequest):
    contexts = retriever.hybrid_search(req.question, 5)
    answer = llm.generate_text(req.question, contexts)
    return {"answer": answer, "contexts": contexts}

class Feedback(BaseModel):
    question: str
    answer: str
    helpful: bool

@app.post("/feedback")
def save_feedback(data: Feedback):
    with open("feedback.log", "a") as f:
        f.write(f"{data.question}|{data.answer}|{data.helpful}\n")
    return {"status": "saved"}

