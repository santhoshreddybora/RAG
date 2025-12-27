from dataclasses import dataclass
from typing import Dict
import hashlib


@dataclass
class RawDocument:
    """
    Final standard format for every document
    """
    id: str
    text: str
    metadata: Dict

    def content_hash(self) -> str:
        """
        Used later for deduplication in vector DB
        """
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

@dataclass
class Chunk:
    text: str
    metadata: Dict

    @property
    def id(self)->str:
        return hashlib.sha256(self.text.encode('utf-8')).hexdigest()
    
    def content_hash(self) -> str:
        return self.id


