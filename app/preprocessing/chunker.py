from typing import List
from app.logger import logging
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.dataclasses import RawDocument,Chunk
from app.preprocessing.deduplicator import ChunkDeduplicator

class DocumentChunker:
    """This class used for chunking the data which we loaded into docs 
    """
    def __init__(self,chunk_size : int=800,chunk_overlap:int =150):
        try:
            self.splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[" ",".","\n","\n\n"])
        except Exception as e:
            logging.error(f"Error in initializing the documentchunker : {e}")
    
    def chunk_documents(self,docs:List[RawDocument])->List[Chunk]:
        try:
            logging.info(f"This ia chunk_document function where all chunking take place")
            all_chunks:List[Chunk]=[]
            for doc in docs:
                splits=self.splitter.split_text(doc.text)
                for index ,text in enumerate(splits):
                    chunk=Chunk(
                        text=text,
                        metadata={
                            **doc.metadata,
                            "chunk_index":index
                        }
                    )
                    all_chunks.append(chunk)
            return all_chunks
        except Exception as e:
            logging.info(f"Error in chunk_document function {e}")
    

    def initiate_document_chunking(self,docs):
        try:
            logging.info("Initiated Document chunking....")
            chunks=self.chunk_documents(docs)
            logging.info(f"Chunks are {chunks}")
            logging.info(f"Total chunks before dedup: {len(chunks)}")
            cd=ChunkDeduplicator()
            unique_chunks=cd.deduplicate(chunks)
            return unique_chunks
        except Exception as e:
            logging.info(f"Error in initiate_document_chunking {e}")

