from app.core.config import settings
from app.ingestion.loader import Documentloader
from app.preprocessing.chunker import DocumentChunker
from app.logger import logging
from app.retrieval.pinecone_manager import PineconeManager
from app.retrieval.bm25 import BM25Manager
from app.retrieval.hybrid_retriever import HybridRetriever
from app.generator.gpt_client import GPTClient
from tests.evaluate_metrics import EvaluateMetrics

def main():
    # print("=== Clinical RAG – Config Test ===")
    # print("OPENAI_API_KEY present:", bool(settings.OPENAI_API_KEY))
    # print("PINECONE_API_KEY present:", bool(settings.PINECONE_API_KEY))
    # print("PINECONE_ENV:", settings.PINECONE_ENV)
    # print("PINECONE_INDEX_NAME:", settings.PINECONE_INDEX_NAME)
    # print("AZURE_BLOB_CONTAINER:", settings.AZURE_BLOB_CONTAINER)
    loader=Documentloader(data_path='data')
    docs=loader.start_data_loading()
    chunker=DocumentChunker(chunk_size=300,chunk_overlap=120)
    unique_chunks=chunker.initiate_document_chunking(docs)
    logging.info(f"Unique chunks count from documents {len(unique_chunks)}")
    pinecone=PineconeManager()
    pinecone.initiate_embeddings(unique_chunks)
    logging.info(" Stored in Pinecone")
    logging.info(f"Pinecone Stats:{pinecone.get_index_stats()}")
    bm25 = BM25Manager()
    bm25.build_index(unique_chunks)
    bm25.save()

    evaluator = EvaluateMetrics()
    results = evaluator.evaluate_all()

    print("\nFinal Metrics:\n", results)

    # query="The Indian Health sector consists of ?"
    # retriever=HybridRetriever()
    # contexts=retriever.hybrid_search(query)
    # if not contexts:
    #     print("❌ No relevant context found")
    #     contexts = ["No relevant context available."]
    # logging.info("Retrieved contexts:")
    # for c in contexts:
    #     logging.info(c)
    #     logging.info("--"*40)
    # llm=GPTClient()
    # answer=llm.generate_text(query,contexts)
    # logging.info("✅ Text generation completed.")
    # logging.info(f"Answer: {answer}")
    # print(" BM25 index stored as bm25_index.pkl")



# if __name__ == "__main__":
#     evaluator = EvaluateMetrics()
#     results = evaluator.evaluate_all()

#     print("\nFinal Metrics:\n", results)



if __name__ == "__main__":
    main()

