from sentence_transformers import CrossEncoder
from app.logger import logging
from typing import List

class CrossEncoderReranker:
    def __init__(self):
        self.model=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def rerank(self,query:str,texts:List[str],top_k:int=5)->List[str]:
        try:
            logging.info("Reranking started in rerank function of CrossEncoder class")
            if not texts:
                return []
            pairs=[(query,text) for text in texts]

            scores = self.model.predict(pairs)

        # Return (text, score) tuples
            return list(zip(texts, scores))
        except Exception as e:
            logging.info(f"Error in rerank function of CrossEncoder class : {e}")