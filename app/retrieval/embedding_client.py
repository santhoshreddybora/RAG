import requests,os
import numpy as np
from typing import List
from app.logger import logging
from dotenv import load_dotenv
load_dotenv()

class EuriEmbeddingClient:
    """
    Wrapper around EURI Embedding API
    """
    def __init__(self):
        self.url=os.getenv('EURI_API_URL')
        self.headers={
            "Content-Type":"application/json",
            "Authorization":f"Bearer {os.getenv('OPENAI_API_KEY')}"
        }
    
    def embed(self,texts:List[str])->List[list]:
        """
        Takes a list of texts and returns list of embeddings
        """
        try:
            logging.info("embeddings started and it is embed fucntion ")
            payload={
                "input": texts,
                "model": "text-embedding-3-small"
            }

            response=requests.post(self.url,headers=self.headers,json=payload)
            if response.status_code != 200:
                raise Exception(f"Embedding API failed: {response.text}")
            
            data=response.json()
            embeddings=[item["embedding"] for item in data['data']]
            logging.info("Embedding done successfully")
            return embeddings
        except Exception as e:
            logging.info(f"Error in embed function {e}")
        

