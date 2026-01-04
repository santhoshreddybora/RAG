import requests,os
import numpy as np
from typing import List
from app.logger import logging
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
# from requests.packages.urllib3.util.retry import Retry
from urllib3.util.retry import Retry
import time
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
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
    
    def embed(self,texts:List[str],retries:int=3)->List[List[float]]:
        """
        Takes a list of texts and returns list of embeddings
        """
        if not texts or not texts[0].strip():
            logging.error("Empty text passed to embed()")
            return []
        for attempt in range(1,retries+1):
            try:
                logging.info("embeddings started and it is embed fucntion ")
                payload={
                    "input": texts,
                    "model": "text-embedding-3-small"

                }
                response=self.session.post(self.url,headers=self.headers,json=payload,
                                    timeout=(3,10))
                if response.status_code !=200:
                    raise Exception(f"Embedding api failed {response.text}")
                response.raise_for_status()
                data=response.json()
                embeddings = [d["embedding"] for d in data.get("data", [])]

                if not embeddings:
                    logging.error("Embedding API returned empty embeddings")
                    return []
                return embeddings
            except requests.exceptions.Timeout:
                logging.error("Embedding API timeout")
                time.sleep(0.5*attempt)
            except requests.exceptions.ConnectionError as e:
                logging.error(f"Embedding connection error: {e}")
                time.sleep(0.5*attempt)
            except Exception as e:
                logging.error(f"Embedding failed: {e}")
                break
        return []
        

