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
                pool_maxsize=20,
                max_retries=3  # Add retries at adapter level
            )
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)

            logging.info("GPTClient initialized successfully")

        except Exception as e:
            logging.error(f"Error in GPTClient init: {e}")

    def generate_text(self, query: str, contexts: list, history: list, retries: int = 3) -> str:
        """Generate text with improved retry logic"""
        
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

        # Retry loop with exponential backoff
        for attempt in range(1, retries + 1):
            try:
                logging.info(f"LLM request attempt {attempt}")

                response = self.client.generate_completion(
                    prompt=prompt,
                    temperature=0.2,
                    max_tokens=600,
                )

                answer = response["choices"][0]["message"]["content"]
                logging.info(f"✅ LLM generated {len(answer)} characters")
                return answer

            except requests.exceptions.Timeout as e:
                wait_time = 2 ** attempt  # 2s, 4s, 8s
                logging.warning(f"⏱️  LLM timeout (attempt {attempt}/{retries})")
                if attempt < retries:
                    logging.info(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error("❌ Request timed out after all retries")
                    return "Sorry, the response is taking too long. Please try again with a shorter question."

            except (requests.exceptions.ConnectionError, 
                    ConnectionResetError,
                    requests.exceptions.ChunkedEncodingError,
                    BrokenPipeError,
                    OSError) as e:
                wait_time = 2 ** attempt  # Exponential backoff
                error_msg = str(e)[:100]
                logging.warning(f"🔌 LLM connection error (attempt {attempt}/{retries}): {error_msg}")
                
                if attempt < retries:
                    logging.info(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error("❌ Connection failed after all retries")
                    return "Sorry, I'm having trouble connecting to the AI service. Please try again in a moment."

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if hasattr(e, 'response') else None
                
                # Handle rate limiting specially
                if status_code == 429:
                    wait_time = 5 * attempt  # Longer wait for rate limits
                    logging.warning(f"🚦 Rate limit hit (attempt {attempt}/{retries})")
                    if attempt < retries:
                        logging.info(f"⏳ Waiting {wait_time}s for rate limit...")
                        time.sleep(wait_time)
                    else:
                        return "Sorry, too many requests. Please wait a moment and try again."
                else:
                    logging.error(f"❌ LLM HTTP error {status_code}: {e}")
                    if attempt < retries:
                        time.sleep(2 ** attempt)
                    else:
                        return "Sorry, AI service error. Please try again later."

            except KeyError as e:
                # Handle malformed API response
                logging.error(f"❌ Malformed API response: {e}")
                if attempt < retries:
                    time.sleep(2 ** attempt)
                else:
                    return "Sorry, received invalid response from AI service."

            except Exception as e:
                logging.error(f"❌ LLM unexpected error (attempt {attempt}/{retries}): {type(e).__name__}: {str(e)[:200]}")
                if attempt < retries:
                    time.sleep(2 ** attempt)
                else:
                    return "Sorry, an unexpected error occurred. Please try again."

        # Should never reach here, but just in case
        logging.error("LLM generation failed after all retries")
        return "Sorry, I'm having trouble right now. Please try again in a few moments."

    def summarize(self, prompt: str, retries: int = 3) -> str:
        """Summarize conversation history with retry logic"""
        
        for attempt in range(1, retries + 1):
            try:
                logging.info(f"Summarization attempt {attempt}/{retries}")
                
                response = self.client.generate_completion(
                    prompt=prompt,
                    temperature=0.2,
                    max_tokens=300
                )
                
                summary = response["choices"][0]["message"]["content"]
                logging.info(f"✅ Summary generated ({len(summary)} chars)")
                return summary
                
            except (requests.exceptions.ConnectionError, 
                    ConnectionResetError,
                    requests.exceptions.ChunkedEncodingError,
                    BrokenPipeError,
                    OSError) as e:
                wait_time = 2 ** attempt  # 2s, 4s, 8s
                error_msg = str(e)[:100]
                logging.warning(f"🔌 Summarization connection error (attempt {attempt}/{retries}): {error_msg}")
                
                if attempt < retries:
                    logging.info(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error("❌ Summarization failed after all retries")
                    return "Unable to generate summary due to connection issues."
            
            except Exception as e:
                logging.error(f"❌ Summarization error (attempt {attempt}/{retries}): {type(e).__name__}: {str(e)[:200]}")
                if attempt < retries:
                    time.sleep(2 ** attempt)
                else:
                    return "Unable to generate summary."
        
        return "Unable to generate summary."
        
    def generate_title(self, question: str, retries: int = 3) -> str:
        """Generate a concise title for the chat session with retry logic"""
        
        prompt = f"""Generate a short, descriptive title (3-6 words) for a chat that starts with this question:

                    "{question}"

                    Requirements:
                    - 3-6 words maximum
                    - Capture the main topic
                    - No quotes, no extra text
                    - Professional and clear

                    Title:"""
        
        for attempt in range(1, retries + 1):
            try:
                logging.info(f"Title generation attempt {attempt}/{retries}")
                
                response = self.client.generate_completion(
                    prompt=prompt,
                    temperature=0.3,
                    max_tokens=20
                )
                
                title = response["choices"][0]["message"]["content"].strip()
                
                # Validation: fallback if AI gives weird response
                if len(title) > 60 or len(title) < 3 or '\n' in title:
                    title = self._extract_simple_title(question)
                
                logging.info(f"✅ Title generated: {title}")
                return title
            
            except (requests.exceptions.ConnectionError, 
                    ConnectionResetError,
                    requests.exceptions.ChunkedEncodingError,
                    BrokenPipeError,
                    OSError) as e:
                wait_time = 2 ** attempt
                error_msg = str(e)[:100]
                logging.warning(f"🔌 Title generation connection error (attempt {attempt}/{retries}): {error_msg}")
                
                if attempt < retries:
                    logging.info(f"⏳ Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error("❌ Title generation failed after all retries, using fallback")
                    return self._extract_simple_title(question)
            
            except Exception as e:
                logging.error(f"❌ Title generation error: {type(e).__name__}: {str(e)[:200]}")
                return self._extract_simple_title(question)
        
        # Fallback if all retries fail
        return self._extract_simple_title(question)
    
    def _extract_simple_title(self, question: str) -> str:
        """Fallback: extract simple title from question"""
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