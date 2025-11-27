from app.retrieval.hybrid_retriever import HybridRetriever

r = HybridRetriever()

res = r.hybrid_search("The Indian Health sector consists of", 10)

print("\n---- SIMPLE RETRIEVAL TEST ----")
print(res)
print(type(res))
