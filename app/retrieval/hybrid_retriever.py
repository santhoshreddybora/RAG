from typing import List
from app.retrieval.bm25 import BM25Manager
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.pinecone_manager import PineconeManager
from app.retrieval.embedding_client import EuriEmbeddingClient
from app.core.config import settings
from app.logger import logging
from app.tracking.mlflow_manager import MLflowManager
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

class HybridRetriever:
    def __init__(self):
        try:
            logging.info("Initializing HybridRetriever")
            self.bm25=BM25Manager()
            self.bm25.load("bm25_index.pkl")
            self.pinecone=PineconeManager()
            self.embedder=EuriEmbeddingClient()
            self.reranker=CrossEncoderReranker()
        except Exception as e:
            logging.error(f"Error initializing HybridRetriever: {e}")

    async def hybrid_search(
                self,
                query: str,
                query_embedding: list,
                top_k: int = 5
            ) -> List[str]:

        try:
            loop = asyncio.get_running_loop()
            logging.info(f"Starting hybrid search in HybridRetriever with query: {query}")

            # 1️⃣ Parallel BM25 + Pinecone
            bm25_task = loop.run_in_executor(
                executor, self.bm25.search, query, top_k
            )

            pinecone_task = loop.run_in_executor(
                executor,
                lambda: self.pinecone.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=self.pinecone.namespace
                )
            )

            bm25_results, pinecone_results = await asyncio.gather(
                bm25_task, pinecone_task
            )

            # 2️⃣ Normalize IDs (🔥 CRITICAL FIX 🔥)
            bm25_ids = [str(doc_id) for doc_id, _ in bm25_results]

            pinecone_ids = [
                str(match.id)
                for match in pinecone_results.matches
                if match.id is not None
            ]

            all_ids = list(set(bm25_ids + pinecone_ids))
            # If top result score is strong, skip rerank
            matches = pinecone_results.matches
            logging.info(f"score of pinecone results:{matches[0].score}")
            if matches and matches[0].score >= 0.85:
                logging.info("High-confidence Pinecone result, skipping rerank")
                return [
                    m.metadata["text"]
                    for m in matches[:top_k]
                    if m.metadata and "text" in m.metadata
                ]
            if not all_ids:
                logging.warning("No document IDs found from BM25 or Pinecone")
                return []

            # 3️⃣ Fetch documents
            fetched = self.pinecone.fetch_by_ids(all_ids)

            contexts = [
                v["metadata"]["text"]
                for v in fetched.get("vectors", {}).values()
                if v.get("metadata", {}).get("text") and len(v["metadata"]["text"]) > 40
            ]

            if not contexts:
                logging.warning("No valid contexts found after fetch")
                return []

            # ⚠️ Cap before reranking
            contexts = contexts[:6]



            # 4️⃣ Rerank
            reranked = await loop.run_in_executor(
                executor,
                self.reranker.rerank,
                query,
                contexts
            )

            reranked.sort(key=lambda x: x[1], reverse=True)
            final_contexts = [text for text, _ in reranked[:top_k]]

            return final_contexts

        except Exception as e:
            logging.error(f"Error in hybrid_search of HybridRetriever: {e}")
            return []
