from typing import List
from app.retrieval.bm25 import BM25Manager
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.pinecone_manager import PineconeManager
from app.retrieval.embedding_client import EuriEmbeddingClient
from app.core.config import settings
from app.logger import logging

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
            # ----------- 5. RERANK -------------------
            reranked_contexts=self.reranker.rerank(query,contexts,top_k=top_k)
            logging.info(f"Total contexts after rerank: {len(reranked_contexts)}")
            return reranked_contexts
        except Exception as e:
            logging.error(f"Error in hybrid_search of HybridRetriever: {e}")
            return [] 