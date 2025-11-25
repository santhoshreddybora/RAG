from rank_bm25 import BM25Okapi
from typing import List
import pickle
from app.logger import logging
from app.dataclasses import Chunk

class BM25Manager:
    def __init__(self,):
        self.corpus=[]
        self.chunk_ids=[]
        self.bm25=None
    
    def build_index(self,chunks:List[Chunk]):
        try:
            logging.info("building index for the BM25 from thte chunks")
            for chunk in chunks:
                tokens=chunk.text.split()
                self.corpus.append(tokens)
                self.chunk_ids.append(chunk.id)
            self.bm25=BM25Okapi(self.corpus)
            logging.info("BM25 index built successfully")
        except Exception as e:
            logging.info(f"Error in building index of BM25 {e}")
    
    def save(self, file_path: str = "bm25_index.pkl"):
        with open(file_path, "wb") as f:
            pickle.dump((self.bm25, self.chunk_ids), f)

    def load(self, file_path: str = "bm25_index.pkl"):
        with open(file_path, "rb") as f:
            self.bm25, self.chunk_ids = pickle.load(f)

    def search(self, query: str, top_k: int = 5):
        try:
            logging.info("searching with the scores with default K=5 ")
            tokens = query.split()
            scores = self.bm25.get_scores(tokens)

            ranked = sorted(
                zip(self.chunk_ids, scores),
                key=lambda x: x[1],
                reverse=True
            )
            return ranked[:top_k]
        except Exception as e:
            logging.error(f"Error in search function {e}")
    


    