from typing import List
from app.retrieval.bm25 import BM25Manager
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.pinecone_manager import PineconeManager
from app.retrieval.embedding_client import EuriEmbeddingClient
from app.core.config import settings
from app.logger import logging
from app.tracking.mlflow_manager import MLflowManager
import numpy as np
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
    
    def hybrid_search(self,query:str,top_k:int=5)->List[str]:
        try:
            logging.info(f"Starting hybrid search in HybridRetriever with query: {query}")
            # ----------- 1. BM25 SEARCH -------------
            bm25_results=self.bm25.search(query,top_k=top_k)
            bm25_ids=[item[0] for item in bm25_results]

            query_embedding=self.embedder.embed([query])[0]
            pinecone_results=self.pinecone.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            pinecone_ids=[match['id'] for match in pinecone_results['matches']]
            # ----------- 3. MERGE IDS ---------------
            all_ids=list(set(bm25_ids + pinecone_ids))
            # ----------- 4. FETCH TEXT --------------
            response = self.pinecone.fetch_by_ids(all_ids)

            contexts=[]
            if "vectors" in response:
                for _id,data in response["vectors"].items():
                    text=data['metadata'].get('text')
                    if text:
                        contexts.append(text)
            logging.info(f"Total contexts before rerank: {len(contexts)}")
            filtered_contexts = [c for c in contexts if len(c) > 30]
            contexts = filtered_contexts
            q_emb = self.embedder.embed([query])[0]

            filtered = []
            for ctx in contexts:
                c_emb = self.embedder.embed([ctx])[0]
                score = np.dot(q_emb, c_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(c_emb))
                if score > 0.5:
                    filtered.append(ctx)

            contexts = filtered
            reranked = self.reranker.rerank(query, contexts)  
            # -> [(text, score), (text, score), ...]

            # FILTER: keep only high quality text
            filtered = [(t, s) for t, s in reranked if s > 0.35]

            if not filtered:
                logging.warning("No contexts passed score threshold, returning top 1 fallback")
                filtered = sorted(reranked, key=lambda x: x[1], reverse=True)[:1]

            # Sort by best score & keep top_k
            filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_k]

            final_contexts = [t for t, _ in filtered]

            logging.info(f"Total contexts after rerank & filter: {len(final_contexts)}")

            # DEBUG (you can keep this for now)
            for i, c in enumerate(final_contexts):
                logging.info(f"Context {i+1}: {c[:150]}")

            return final_contexts
        except Exception as e:
            logging.error(f"Error in hybrid_search of HybridRetriever: {e}")
            return [] 