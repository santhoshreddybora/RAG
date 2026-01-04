from app.logger import logging
from dotenv import load_dotenv
load_dotenv()

import os
import requests
from app.tracking.mlflow_manager import MLflowManager
import time
from euriai import EuriaiClient



class GPTClient:
    def __init__(self):
        try:
            # ✅ Correct env variable name
            self.url = os.getenv("EURI_CHAT_URI")   # NOT "url"
            self.api_key = os.getenv("OPENAI_API_KEY")

            if not self.url or not self.api_key:
                raise ValueError("EURI_CHAT_URI or OPENAI_API_KEY is missing in .env")

         
            self.client=EuriaiClient(
                api_key=self.api_key,
                model='gpt-4.1-nano'
            )

            logging.info("GPTClient initialized successfully")

        except Exception as e:
            logging.error(f"Error in GPTClient init: {e}")

    def generate_text(self, query: str, contexts: list,history:list,retries:int = 3) -> str:
        
            logging.info("Generating text in generate_text function of GPTClient class")

            # ✅ 1. Important guard
            if not contexts or len(contexts) == 0:
                logging.warning("No contexts available to generate answer.")
                # yield "I do not have enough information in the provided documents."
                return "No contexts available to generate answer."
            
            history_prompt = ""
            for msg in history:
                history_prompt += f"{msg['role']}: {msg['content']}\n"
            
            # ✅ 2. Now join contexts
            context = "\n\n".join(contexts)

            prompt = f"""
                        You are a health care  analyst. Use the context below to answer the question.

                        If a direct answer is not present, summarize the most relevant information 
                        from the context related to the question.

                        Conversation so far:
                        {history_prompt}
                        
                        Context:
                        {context}

                        Question:
                        {query}

                        Give a detailed, sentence answer in a professional tone.

                        If absolutely nothing is found, then respond:
                        "I do not have enough information in the provided documents."
                        """
            for attempt in range(1, retries + 1):
                try:
                    logging.info(f"LLM request attempt {attempt}")

                    response = self.client.generate_completion(
                        prompt=prompt,
                        temperature=0.2,
                        max_tokens=1000,
                    )

                    return response["choices"][0]["message"]["content"]

                except requests.exceptions.ConnectionError as e:
                    logging.warning(
                        f"LLM connection error (attempt {attempt}): {e}"
                    )
                    time.sleep(0.7 * attempt)

                except Exception as e:
                    logging.error(
                        f"LLM unexpected error: {e}", exc_info=True
                    )
                    break

        # 🔴 FINAL failure — no silent continuation
            logging.error("LLM generation failed after retries")
            return "Sorry, I’m having trouble generating a response right now."
        
    def summarize(self, prompt: str) -> str:
        response = self.client.generate_completion(
                prompt=prompt,
                temperature=0.2,
                max_tokens=300
            )
        return response["choices"][0]["message"]["content"]