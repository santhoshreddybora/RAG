# app/config.py

import os
from dotenv import load_dotenv
from app.logger import logging
# Load variables from .env into environment
load_dotenv()

class Settings:
    """
    Simple config holder.
    Reads from environment variables (which are loaded from .env).
    """
    OPENAI_API_KEY: str | None
    PINECONE_API_KEY: str | None
    PINECONE_ENV: str | None
    PINECONE_INDEX_NAME: str | None
    AZURE_BLOB_CONNECTION_STRING: str | None
    AZURE_BLOB_CONTAINER: str | None

    def __init__(self) -> None:
        try:
            logging.info("Checking all keys are been set or not ")
            self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
            self.PINECONE_ENV = os.getenv("PINECONE_ENV")
            self.PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
            self.AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
            self.AZURE_BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER")

            # Simple validation – fail fast if critical things are missing
            missing = []
            if not self.OPENAI_API_KEY:
                missing.append("OPENAI_API_KEY")
            if not self.PINECONE_API_KEY:
                missing.append("PINECONE_API_KEY")
            if not self.PINECONE_INDEX_NAME:
                missing.append("PINECONE_INDEX_NAME")

            if missing:
                print(f"[Settings] WARNING: Missing env vars: {', '.join(missing)}")
        except Exception as e:
            logging.error("Error in config file in Settings class")

# create a singleton settings object you can import everywhere
settings = Settings()
