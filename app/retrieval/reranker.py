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

            scores=self.model.predict(pairs)
            ranked=sorted(zip(texts,scores),key=lambda x:x[1],reverse=True)

            return [text for text,score in ranked[:top_k]]
        except Exception as e:
            logging.info(f"Error in rerank function of CrossEncoder class : {e}")