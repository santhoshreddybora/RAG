import numpy as np
import redis
import json
import hashlib
from typing import Optional
from app.retrieval.embedding_client import EuriEmbeddingClient
from app.logger import logging


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.85):
        self.embedder = EuriEmbeddingClient()
        self.threshold = similarity_threshold

        self.redis = redis.Redis(
            host="localhost",   # ✅ FIXED
            port=6379,
            decode_responses=True
        )

    @staticmethod
    def _cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def get(self, session_id: str, query: str) -> Optional[str]:
        try:
            query_hash = self._hash(query)
            key = f"rag_cache:{session_id}:{query_hash}"

            cached = self.redis.get(key)
            if not cached:
                return None

            cached = json.loads(cached)

            query_emb = self.embedder.embed([query])[0]
            score = self._cosine_similarity(query_emb, cached["embedding"])

            if score >= self.threshold:
                logging.info(f"Semantic cache HIT (score={score:.2f})")
                return cached["answer"]

            logging.info(f"Semantic cache MISS (score={score:.2f})")
            return None

        except Exception as e:
            logging.error(f"SemanticCache.get error: {e}")
            return None

    def set(self, session_id: str, query: str, answer: str):
        try:
            query_hash = self._hash(query)
            key = f"rag_cache:{session_id}:{query_hash}"

            emb = self.embedder.embed([query])[0]

            payload = {
                "embedding": emb,
                "answer": answer
            }

            self.redis.set(
                key,
                json.dumps(payload),
                ex=60 * 60 * 24  # 24 hours TTL
            )

            logging.info("Cached answer in SemanticCache")

        except Exception as e:
            logging.error(f"SemanticCache.set error: {e}")
