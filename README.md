# Clinical RAG – Clinical Question Answering with Hybrid Retrieval

This project is an **end-to-end Retrieval-Augmented Generation (RAG)** system for **clinical / healthcare Q&A**.

It ingests clinical / healthcare PDF documents, chunks and embeds them, stores them in **Pinecone**, builds a **BM25** lexical index, and exposes a **FastAPI** endpoint that uses **hybrid retrieval + cross-encoder reranking** to answer questions. A **Streamlit UI** sits on top of the API.

---

## Key Features (What This Project Implements)

- **End-to-end RAG pipeline for clinical QA**
  - PDF / DOCX / TXT ingestion + optional OCR for PDF images
  - Text cleaning, chunking, and deduplication
- **Hybrid retrieval**
  - Dense retrieval with **Sentence Transformer embeddings via EURI API**
  - Lexical retrieval with **BM25 (rank-bm25)**
  - Merged results, then **cross-encoder reranking**
- **Vector store**
  - **Pinecone** as dense vector DB (1536-dim embeddings)
- **LLM generation**
  - Uses **EURI chat completion API** with `gpt-4.1-nano` (or equivalent)
  - Grounded prompting: uses retrieved contexts only
- **Evaluation + metrics**
  - Evaluation dataset (`evaluation_dataset.json`)
  - Computes:
    - Recall
    - Precision
    - Hallucination rate
    - Faithfulness score (sentence-transformers cosine similarity)
  - Logs metrics to **MLflow** (Azure ML workspace configured)
- **Serving**
  - **FastAPI** backend with `/ask` endpoint (and optional `/feedback`)
  - **Streamlit UI** for interactive QA
- **Deployment**
  - Dockerized **API** and **UI** separately
  - Deployed both to **Azure Web Apps for Containers**, pulling images from **Azure Container Registry (ACR)**

>  GitHub Actions CI/CD and conversational chat history are **not yet integrated**. Those are natural next steps.

---

##  Project Structure (High Level)

```bash
clinical-rag/
├── app/
│   ├── api/
│   │   └── main.py              # FastAPI app (backend API)
│   ├── core/
│   │   └── config.py            # Settings / env config (pydantic / dotenv)
│   ├── dataclasses.py           # RawDocument, Chunk, etc.
│   ├── ingestion/
│   │   └── loader.py            # Documentloader (PDF/TXT/DOCX + OCR)
│   ├── preprocessing/
│   │   └── chunker.py           # DocumentChunker (chunk_size, overlap, cleaning)
│   ├── retrieval/
│   │   ├── bm25.py              # BM25Manager (build/save/load index)
│   │   ├── embedding_client.py  # EuriEmbeddingClient (calls EURI embeddings API)
│   │   ├── pinecone_manager.py  # PineconeManager (index create/upsert/query)
│   │   ├── hybrid_retriever.py  # HybridRetriever (BM25 + Pinecone + rerank)
│   │   └── reranker.py          # CrossEncoderReranker (sentence-transformers)
│   ├── generator/
│   │   └── gpt_client.py        # GPTClient (EURI chat completion API)
│   ├── tracking/
│   │   └── mlflow_manager.py    # MLflowManager (logs metrics/runs)
│   └── logger.py                # Logging configuration
│
├── tests/
│   └── evaluate_metrics.py      # Evaluate recall / precision / hallucination / faithfulness
|   ├──  evaluation_dataset.json
│
├── data/
│   ├──                    # Source PDFs / docs (your clinical docs)
│   ├──   # Small QA evaluation dataset
│   └── ...                      # Other input files
│
├── ui/
│   ├── app.py                   # Streamlit UI
│   └── Dockerfile               # UI Dockerfile (Streamlit)
│
├── bm25_index.pkl               # Serialized BM25 index
├── requirements.txt
├── Dockerfile                   # API Dockerfile (FastAPI)
├── .env                         # Local environment variables (not committed)
└── main.py                      # Orchestration script (ingestion, indexing, sample query)

⚙️ Tech Stack

Language: Python 3.10

Frameworks:

Backend: FastAPI

Frontend: Streamlit

RAG / NLP:

sentence-transformers

rank-bm25

EURI API for embeddings & chat completions

Vector DB: Pinecone

Indexing:

BM25 for lexical search

Hybrid retrieval + cross-encoder reranking

Tracking:

MLflow (pointing to Azure ML workspace)

Deployment:

Docker, Azure Container Registry (ACR)

Azure Web App for Containers (API + UI)

🔑 Environment Variables

Create a .env file at repo root for local dev:

# ---- EURI / LLM ----
OPENAI_API_KEY=your_euri_api_key_here
EURI_EMBED_URI=https://api.euron.one/api/v1/euri/embeddings
EURI_CHAT_URI=https://api.euron.one/api/v1/euri/chat/completions

# ---- Pinecone ----
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=clinical-rag
PINECONE_ENV=us-east-1

# ---- Azure / MLflow (optional, for tracking) ----
MLFLOW_TRACKING_URI=azureml://...
AZUREML_EXPERIMENT_NAME=Clinical-RAG

# ---- Other ----
AZURE_BLOB_CONTAINER=clinical-docs        # if/when used


In Azure Web Apps, these go into Configuration → Application Settings instead of .env.

🏗️ 1. Setup & Installation (Local)
# 1. Clone repo
git clone <your-repo-url>
cd clinical-rag

# 2. Create virtual env (conda or venv)
conda create -n rag python=3.10 -y
conda activate rag

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt


Make sure tesseract is installed on your machine if you use OCR for PDF images.

📥 2. Document Ingestion & Index Building
Step 1: Put documents

Place your PDFs / DOCX / TXT in the data/ (or data/) folder that Documentloader points to.

In main.py you have logic like:

from app.ingestion.loader import Documentloader
from app.preprocessing.chunker import DocumentChunker
from app.retrieval.pinecone_manager import PineconeManager
from app.retrieval.bm25 import BM25Manager
from app.logger import logging

def main():
    loader = Documentloader(data_path="data")  # or "data/raw"
    docs = loader.start_data_loading()

    chunker = DocumentChunker(chunk_size=300, chunk_overlap=120)
    unique_chunks = chunker.initiate_document_chunking(docs)
    logging.info(f"Unique chunks count: {len(unique_chunks)}")

    pinecone = PineconeManager()
    pinecone.initiate_embeddings(unique_chunks)
    logging.info("Stored in Pinecone")
    logging.info(f"Pinecone Stats: {pinecone.get_index_stats()}")

    bm25 = BM25Manager()
    bm25.build_index(unique_chunks)
    bm25.save()
    logging.info("BM25 index stored as bm25_index.pkl")

if __name__ == "__main__":
    main()


Run:

python main.py


This will:

Load docs (with OCR on images where needed)

Clean + chunk + deduplicate text

Generate embeddings via EURI API

Upsert vectors into Pinecone

Build and save BM25 index to bm25_index.pkl

3. Evaluation & Metrics

You have an evaluation dataset (data/evaluation_dataset.json) like:

[
  {
    "question": "The Indian Health sector consists of",
    "relevant_texts": [
      "The Indian health sector comprises various components, including medical care providers such as physicians, specialist clinics, nursing homes, and hospitals.",
      "It also includes diagnostic service centers and pathology laboratories, as well as medical equipment manufacturers. Additionally, the sector encompasses contract research organizations (CROs) and pharmaceutical manufacturers."
    ]
  },
  ...
]


The evaluation code (tests/evaluate_metrics.py) roughly:

Loops over questions

Gets hybrid retrieval results

Calculates semantic similarity between expected text and retrieved chunks

Computes:

Recall

Precision

Hallucination rate (answer content not grounded in context)

Faithfulness score (cosine similarity between answer and context)

Example usage (from main.py):

from tests.evaluate_metrics import EvaluateMetrics

evaluator = EvaluateMetrics()
results = evaluator.evaluate_all()
print("\nFinal Metrics:\n", results)


Example printed metrics:

Final Metrics:
{'recall': 0.5, 'precision': 0.22, 'hallucination_rate': 0.5, 'faithfulness_score': -0.05}


These metrics are also logged to MLflow (if configured).

🌐 4. Running the API Locally (FastAPI)

To run the API:

uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload


Then open:

Swagger UI: http://localhost:8000/docs

Health/test endpoints (if any): http://localhost:8000/

Main endpoint:

POST /ask
{
  "question": "The Indian health sector consists of"
}


Response:

{
  "answer": "...LLM generated answer...",
  "contexts": [
    "retrieved chunk 1 ...",
    "retrieved chunk 2 ..."
  ]
}

🖥️ 5. Running the UI Locally (Streamlit)

From project root:

streamlit run ui/app.py


The UI:

Takes a question as input

Calls the FastAPI /ask endpoint

Displays:

Answer from LLM

Top retrieved source chunks

Allows simple feedback (helpful / not helpful), which is logged to feedback.log

Make sure API_URL or BACKEND_URL in ui/app.py points to:

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

🐳 6. Docker & Azure Deployment (What You Have Now)
API Dockerfile (root Dockerfile)
FROM python:3.10

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


Build & tag:

docker build --no-cache -t clinical-rag-api .
docker tag clinical-rag-api <your-acr>.azurecr.io/clinical-rag-api:v1
docker push <your-acr>.azurecr.io/clinical-rag-api:v1


Then in Azure Web App (API):

Type: Linux, Docker Container

Image source: Azure Container Registry

Image: clinical-rag-api:v1

App settings:

WEBSITES_PORT = 8000

plus all env vars (API keys, Pinecone, etc.)

UI Dockerfile (ui/Dockerfile)
FROM python:3.10

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]


Build & tag:

docker build --no-cache -t clinical-rag-ui -f ui/Dockerfile .
docker tag clinical-rag-ui <your-acr>.azurecr.io/clinical-rag-ui:v1
docker push <your-acr>.azurecr.io/clinical-rag-ui:v1


In Azure Web App (UI):

Type: Linux, Docker Container

Image: clinical-rag-ui:v1

App settings:

WEBSITES_PORT = 8501

BACKEND_URL = https://<your-api-app>.azurewebsites.net

📌 Current Status vs. Future Enhancements

✅ Currently implemented:

E2E RAG pipeline (ingestion → indexing → hybrid retrieval → rerank → answer)

Basic evaluation metrics with MLflow logging

FastAPI backend with /ask endpoint

Streamlit UI (question, answer, sources, feedback)

Manual Docker build & Azure App Service deployment (API + UI)

🚧 Planned / Next steps (not yet done):

Conversational chat with chat history (stateful sessions)

CI/CD with GitHub Actions (auto build & deploy on dev / main)

More robust evaluation (bigger dataset, better metrics dashboards)

Authentication / authorization for clinical use (Azure AD, etc.)

Centralized logging & monitoring via Application Insights

🧾 License

(Choose one for your repo: MIT, Apache 2.0, etc.)

MIT License
...


If you’re reading this in the repo, start with:

conda activate rag
python main.py   # index docs
uvicorn app.api.main:app --reload
streamlit run ui/app.py
