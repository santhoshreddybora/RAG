from pinecone import Pinecone,ServerlessSpec
from typing import List
from app.dataclasses import Chunk
from dotenv import load_dotenv
load_dotenv()
import os
from app.logger import logging
from app.retrieval.embedding_client import EuriEmbeddingClient

class PineconeManager:
    def __init__(self):
        try:
            self.pc=Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            self.index_name=os.getenv('PINECONE_INDEX_NAME')

            if self.index_name not in [i["name"] for i in self.pc.list_indexes()]:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region="us-east-1"
                    )
                )
            self.index=self.pc.Index(self.index_name)
        except Exception as e:
            logging.error(f"ERROR in init block of pinecone manager{e}")
    
    def upsert_chunks(self,chunks:List[Chunk],embeddings:List[list]):
        try:
            logging.info("upserting chunks into pinecone db through upsert_chunks function")
            vectors=[]
            for chunk,vector in zip(chunks,embeddings):
                vectors.append(
                    (chunk.id,
                    vector,
                    {
                        **chunk.metadata,
                        "text":chunk.text
                    })
                )
            self.index.upsert(vectors=vectors)
            logging.info("embedding inserted into pinecone db")
        except Exception as e:
            logging.error(f"Error inserting into pinecone {e}")


    def get_index_stats(self):
        return self.index.describe_index_stats()
    
    def fetch_by_ids(self, ids: list):
        """Fetch text/metadata from Pinecone using chunk IDs"""
        result = self.index.fetch(ids=ids)
        return result

    def initiate_embeddings(self,unique_chunks):
        try:
            logging.info("Initiating the embeddings and insert vectors into pinecone db")
            embedder=EuriEmbeddingClient()
            chunk_text=[chunk.text for chunk in unique_chunks]
            embeddings=embedder.embed(chunk_text)
            print(f"Embeddings generated: {len(embeddings)}")
            pinecone=PineconeManager()
            pinecone.upsert_chunks(unique_chunks,embeddings=embeddings)
            logging.info("Initiate embeddings is completed ")
        except Exception as e:
            logging.error(f"Error in initiate_embeddings {e}")



