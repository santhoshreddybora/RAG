from typing import List,Set
from app.dataclasses import Chunk
from app.logger import logging

class ChunkDeduplicator:
    def __init__(self):
        self.seen_hashes:Set[str]=set()
    
    def deduplicate(self,chunks:List[str])->List[Chunk]:
        try:
            logging.info("Deduplicating the chunks in deduplicate fucntion")   
            unique_chunks=[]
            for chunk in chunks:
                h=chunk.content_hash()

                if h not in self.seen_hashes:
                    self.seen_hashes.add(h)
                    unique_chunks.append(chunk)
            return unique_chunks
        except Exception as e:
            logging.info(f"Error in deduplicate function in ChunkDeduplicator class :{e}")
