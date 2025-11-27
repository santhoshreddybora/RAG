from app.tracking.mlflow_manager import MLflowManager
import json
from app.retrieval.hybrid_retriever import HybridRetriever
from app.logger import logging
from app.generator.gpt_client import GPTClient
from sentence_transformers import SentenceTransformer, util


class EvaluateMetrics:

    def __init__(self):
        self.mlflow = MLflowManager()
        self.retriever = HybridRetriever()
        self.llm = GPTClient()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        with open("tests/evaluation_dataset.json") as f:
            self.dataset = json.load(f)

    # ----------------- UTIL FUNCTIONS --------------------

    def is_hallucinated(self, answer, contexts):
        if not contexts:
            return True 
        context = " ".join(contexts)
        answer_emb = self.model.encode(answer)
        context_emb = self.model.encode(context)
        sim = util.cos_sim(answer_emb, context_emb)[0][0]

        # if similarity is very low -> hallucination
        if sim < 0.4:
            return True
        return False
    # ----------------- MAIN FUNCTIONS --------------------

    def calculate_recall_and_precision(self):
        print("DATASET SIZE:", len(self.dataset))
        print("FIRST ROW:", self.dataset[0])
        logging.info(f"Dataset size:{len(self.dataset)}")
        logging.info(f"First row:{self.dataset[0]}")
                     
        total_relevant = 0
        retrieved_relevant = 0
        retrieved_total = 0

        for row in self.dataset:
            question = row["question"]
            expected = row["relevant_texts"]
            

            results = self.retriever.hybrid_search(question, 5)
            print(f"Results from hybrid search: {results}")
            if not results or len(results)==0:
                continue
            print("\nQUESTION:", question)
            print("TOP CONTEXTS FROM VECTOR DB:")

            for i, c in enumerate(results[:3]):
                print(f"{i+1}. {c[:200]}")
            retrieved_total += len(results)

            result_embeddings = self.model.encode(results)

            for exp in expected:
                total_relevant += 1
                exp_embedding = self.model.encode(exp)

                similarity_scores = util.cos_sim(exp_embedding, result_embeddings)[0]
                print("SIMILARITY in precision and recall:", similarity_scores)

                max_score = max(similarity_scores)

                # If semantic similarity > threshold → consider found
                if max_score > 0.65:
                    retrieved_relevant += 1

        recall = retrieved_relevant / total_relevant if total_relevant else 0
        precision = retrieved_relevant / retrieved_total if retrieved_total else 0

        return recall, precision

    def calculate_hallucination_rate(self):
        hallucinations = 0
        answered=0
        for row in self.dataset:
            question = row["question"]

            contexts = self.retriever.hybrid_search(question, 5)
            if not contexts:
                continue
            answer = self.llm.generate_text(question, contexts)
            answered+=1

            if self.is_hallucinated(answer, contexts):
                hallucinations += 1

        hallucination_rate = hallucinations / answered if answered != 0 else 0
        return hallucination_rate

    def calculate_faithfulness(self):
        all_scores = []

        for row in self.dataset:
            question = row["question"]

            contexts = self.retriever.hybrid_search(question, 5)
            if not contexts or len(contexts)<1:
                continue
            answer = self.llm.generate_text(question, contexts)

            context_text = " ".join(contexts)

            answer_vec = self.model.encode(answer)
            context_vec = self.model.encode(context_text)

            similarity = util.cos_sim(answer_vec, context_vec)
            print("SIMILARITY:", similarity)
            all_scores.append(similarity)

        faithfulness_score =float(similarity.mean())
        return faithfulness_score

    # ----------------- MASTER FUNCTION --------------------
    import mlflow
    if mlflow.active_run():
        mlflow.end_run()
    def evaluate_all(self):
        try:
            logging.info("Starting all evaluations")
            self.mlflow.start_run("Full RAG Evaluation")

            recall, precision = self.calculate_recall_and_precision()
            hallucination_rate = self.calculate_hallucination_rate()
            faithfulness = self.calculate_faithfulness()

            self.mlflow.log_metric("recall", recall)
            self.mlflow.log_metric("precision", precision)
            self.mlflow.log_metric("hallucination_rate", hallucination_rate)
            self.mlflow.log_metric("faithfulness_score", faithfulness)

            self.mlflow.end_run()

            logging.info(" All evaluations completed")

            return {
                "recall": recall,
                "precision": precision,
                "hallucination_rate": hallucination_rate,
                "faithfulness_score": faithfulness
            }

        except Exception as e:
            logging.error(f"Error in evaluate_all: {e}")
