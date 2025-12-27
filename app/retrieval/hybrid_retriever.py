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

    async def hybrid_search(self,query:str,top_k:int=5)->List[str]:
        try:
            logging.info(f"Starting hybrid search in HybridRetriever with query: {query}")
            loop = asyncio.get_running_loop()

            # 1️⃣ Parallel BM25 + Query Embedding
            bm25_task = loop.run_in_executor(
                executor, self.bm25.search, query, top_k
            )

            embed_task = loop.run_in_executor(
                executor, self.embedder.embed, [query]
            )

            bm25_results, query_embedding = await asyncio.gather(
                bm25_task, embed_task
            )

            # 2️⃣ Pinecone vector search (executor)
            pinecone_results = await loop.run_in_executor(
                executor,
                lambda: self.pinecone.index.query(
                    vector=query_embedding[0],
                    top_k=top_k,
                    include_metadata=True,
                    namespace=self.pinecone.namespace
                )
            )

            # 3️⃣ Merge IDs
            bm25_ids = [x[0] for x in bm25_results]
            pinecone_ids = [m["id"] for m in pinecone_results["matches"]]
            all_ids = list(set(bm25_ids + pinecone_ids))

            # 4️⃣ Fetch chunks
            response = self.pinecone.fetch_by_ids(all_ids)

            contexts = [
                v["metadata"]["text"]
                for v in response.get("vectors", {}).values()
                if len(v["metadata"].get("text", "")) > 30
            ]

            if not contexts:
                return []

            # 5️⃣ Rerank (executor — VERY IMPORTANT)
            reranked = await loop.run_in_executor(
                executor,
                self.reranker.rerank,
                query,
                contexts
            )

            # 6️⃣ Filter & return
            reranked = sorted(reranked, key=lambda x: x[1], reverse=True)
            final_contexts = [t for t, _ in reranked[:top_k]]

            return final_contexts
        except Exception as e:
            logging.error(f"Error in hybrid_search of HybridRetriever: {e}")
            return [] 