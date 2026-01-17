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
            self.url = os.getenv("EURI_CHAT_URI")
            self.api_key = os.getenv("OPENAI_API_KEY")

            if not self.url or not self.api_key:
                raise ValueError("EURI_CHAT_URI or OPENAI_API_KEY is missing in .env")

            self.client = EuriaiClient(
                api_key=self.api_key,
                model='gpt-4.1-nano'
            )
            
            # Configure session for connection pooling
            self.session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=20
            )
            self.session.mount("https://", adapter)

            logging.info("GPTClient initialized successfully")

        except Exception as e:
            logging.error(f"Error in GPTClient init: {e}")

    def generate_text(self, query: str, contexts: list, history: list, retries: int = 2) -> str:
        
        logging.info("Generating text in generate_text function of GPTClient class")

        if not contexts or len(contexts) == 0:
            logging.warning("No contexts available to generate answer.")
            return "No contexts available to generate answer."
        
        # Build history prompt
        history_prompt = ""
        for msg in history:
            history_prompt += f"{msg['role']}: {msg['content']}\n"
        
        # Join contexts - limit context size
        context = "\n\n".join(contexts[:5])  # Use only top 5 contexts
        
        # Detect formatting requirements
        query_lower = query.lower()
        formatting_instructions = ""
        
        if any(word in query_lower for word in ['points', 'list', 'bullet', 'list down']):
            formatting_instructions = """
FORMATTING: Use bullet points (•)"""
        
        elif any(word in query_lower for word in ['table', 'tabular', 'format as table', 'in table format']):
            formatting_instructions = """
FORMATTING: Create table with pipes (|) - keep columns SHORT:
Column1 | Column2 | Column3
Data1 | Data2 | Data3"""

        # Shorter system prompt
        system_instructions = """You are a healthcare analyst.

                                Rules:
                                - Answer from context only
                                - Be concise and accurate
                                - Follow requested format (bullet/table/paragraph)
                                - For tables: Use | separators, SHORT column names"""

        prompt = f"""{system_instructions}

                        {formatting_instructions}

                        History: {history_prompt[:500]}

                            Context: {context[:2000]}

                            Q: {query}

                            Answer:"""

        for attempt in range(1, retries + 1):
            try:
                logging.info(f"LLM request attempt {attempt}")

                response = self.client.generate_completion(
                    prompt=prompt,
                    temperature=0.2,
                    max_tokens=600,  # Reduced for faster response
                )

                answer = response["choices"][0]["message"]["content"]
                logging.info(f"✅ LLM generated {len(answer)} characters")
                return answer

            except requests.exceptions.Timeout:
                logging.warning(f"⏱️  LLM timeout (attempt {attempt})")
                if attempt == retries:
                    return "Sorry, the response is taking too long. Please try again."
                time.sleep(0.3)

            except (requests.exceptions.ConnectionError, 
                    ConnectionResetError, 
                    requests.exceptions.ChunkedEncodingError) as e:
                logging.warning(f"🔌 LLM connection error (attempt {attempt}): {str(e)[:100]}")
                if attempt == retries:
                    return "Sorry, connection error. Please try a shorter question."
                time.sleep(0.5 * attempt)

            except requests.exceptions.HTTPError as e:
                logging.error(f"❌ LLM HTTP error: {e}")
                if attempt == retries:
                    return "Sorry, AI service error. Please try again."
                time.sleep(0.3)

            except Exception as e:
                logging.error(f"❌ LLM unexpected error: {e}")
                return "Sorry, an error occurred. Please try again."

        logging.error("LLM generation failed after retries")
        return "Sorry, I'm having trouble right now. Please try again."
    def summarize(self, prompt: str) -> str:
        """Summarize conversation history"""
        response = self.client.generate_completion(
            prompt=prompt,
            temperature=0.2,
            max_tokens=300
        )
        return response["choices"][0]["message"]["content"]
    
    def generate_title(self, question: str) -> str:
        """Generate a concise title for the chat session"""
        prompt = f"""Generate a short, descriptive title (3-6 words) for a chat that starts with this question:

"{question}"

Requirements:
- 3-6 words maximum
- Capture the main topic
- No quotes, no extra text
- Professional and clear

Title:"""
        
        try:
            response = self.client.generate_completion(
                prompt=prompt,
                temperature=0.3,
                max_tokens=20
            )
            
            title = response["choices"][0]["message"]["content"].strip()
            
            # Validation: fallback if AI gives weird response
            if len(title) > 60 or len(title) < 3 or '\n' in title:
                # Simple fallback: extract keywords
                title = self._extract_simple_title(question)
            
            return title
            
        except Exception as e:
            logging.error(f"Title generation failed: {e}")
            return self._extract_simple_title(question)
    
    def _extract_simple_title(self, question: str) -> str:
        """Fallback: extract simple title from question"""
        # Remove common question words
        words_to_remove = ['what', 'how', 'why', 'when', 'where', 'is', 'are', 
                          'can', 'could', 'would', 'tell me', 'explain', 'describe',
                          'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for']
        
        title = question.lower()
        for word in words_to_remove:
            title = title.replace(f' {word} ', ' ')
        
        # Clean up and capitalize
        title = ' '.join(title.split())[:50].strip()
        title = ' '.join(word.capitalize() for word in title.split())
        
        # If still too short, use original
        if len(title) < 10:
            title = question[:50]
        
        return title