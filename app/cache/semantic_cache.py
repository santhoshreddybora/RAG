import numpy as np
import redis
import json
import hashlib
from typing import Optional
from app.retrieval.embedding_client import EuriEmbeddingClient
from app.logger import logging

class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.88):  # Changed from 0.85 to 0.88
        self.embedder = EuriEmbeddingClient()
        self.threshold = similarity_threshold

        self.redis = redis.Redis(
            host="localhost",
            port=6379,
            decode_responses=True
        )

    @staticmethod
    def _cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def get(self, session_id: str, query: str, query_embedding: list) -> Optional[str]:
        try:
            # Search ALL cached questions for this session
            pattern = f"rag_cache:{session_id}:*"
            keys = self.redis.keys(pattern)
            
            if not keys:
                return None
            
            best_score = 0
            best_answer = None
            
            # Check similarity against all cached questions in this session
            for key in keys:
                cached = self.redis.get(key)
                if not cached:
                    continue
                    
                cached_data = json.loads(cached)
                score = self._cosine_similarity(query_embedding, cached_data["embedding"])
                
                if score > best_score:
                    best_score = score
                    best_answer = cached_data["answer"]
            
            # Return if above threshold
            if best_score >= self.threshold:
                logging.info(f"Semantic cache HIT (score={best_score:.2f})")
                return best_answer
            
            logging.info(f"Semantic cache MISS (best score={best_score:.2f})")
            return None
            
        except Exception as e:
            logging.error(f"SemanticCache.get error: {e}")
            return None

    def set(self, session_id: str, query: str, answer: str, query_embedding: list):
        try:
            query_hash = self._hash(query)
            key = f"rag_cache:{session_id}:{query_hash}"

            payload = {
                "embedding": query_embedding,
                "answer": answer,
                "query": query  # Added: Store original query for debugging
            }

            self.redis.set(
                key,
                json.dumps(payload),
                ex=60 * 60 * 24  # 24 hours TTL
            )

            logging.info("Cached answer in SemanticCache")

        except Exception as e:
            logging.error(f"SemanticCache.set error: {e}")