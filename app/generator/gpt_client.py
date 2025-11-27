from app.logger import logging
from dotenv import load_dotenv
load_dotenv()

import os
import requests
from app.tracking.mlflow_manager import MLflowManager
import time



class GPTClient:
    def __init__(self):
        try:
            # ✅ Correct env variable name
            self.url = os.getenv("EURI_CHAT_URI")   # NOT "url"
            self.api_key = os.getenv("OPENAI_API_KEY")

            if not self.url or not self.api_key:
                raise ValueError("EURI_CHAT_URI or OPENAI_API_KEY is missing in .env")

            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            self.mlflow=MLflowManager()
            logging.info("GPTClient initialized successfully")

        except Exception as e:
            logging.error(f"Error in GPTClient init: {e}")

    def generate_text(self, query: str, contexts: list) -> str:
        try:
            logging.info("Generating text in generate_text function of GPTClient class")

            # ✅ 1. Important guard
            if not contexts or len(contexts) == 0:
                logging.warning("No contexts available to generate answer.")
                return "I do not have enough information in the provided documents."

            # ✅ 2. Now join contexts
            context = "\n\n".join(contexts)

            prompt = f"""
                        You are a clinical healthcare analyst. Use the context below to answer the question.

                        If a direct answer is not present, summarize the most relevant information 
                        from the context related to the question. DO NOT make up answers. If the context does not contain relevant information,

                        Context:
                        {context}

                        Question:
                        {query}

                        Give a detailed, 5–8 sentence answer in a professional tone.

                        If absolutely nothing is found, then respond:
                        "I do not have enough information in the provided documents."
                        """

            payload = {
                "model": "gpt-4.1-nano",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a factual, safe medical assistant"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.2
            }

            response = requests.post(self.url, headers=self.headers, json=payload)

            if response.status_code != 200:
                logging.error(f"GPT API failed: {response.text}")
                return "Error occurred while generating answer."

            data = response.json()
            self.mlflow.log_param("model", "gpt-4.1-nano")
            self.mlflow.log_param("temperature", 0.2)
            start = time.time()
            answer = data["choices"][0]["message"]["content"]
            gen_time = time.time() - start

            self.mlflow.log_metric("generation_time", gen_time)
            self.mlflow.log_text(answer, "answer.txt")

            return data["choices"][0]["message"]["content"]

        except Exception as e:
            logging.error(f"Error in generate_text: {e}")
            return "Error occurred while generating answer."
