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

            logging.info("GPTClient initialized successfully")

        except Exception as e:
            logging.error(f"Error in GPTClient init: {e}")

    def generate_text(self, query: str, contexts: list, history: list, retries: int = 3) -> str:
        
        logging.info("Generating text in generate_text function of GPTClient class")

        # Guard against empty contexts
        if not contexts or len(contexts) == 0:
            logging.warning("No contexts available to generate answer.")
            return "No contexts available to generate answer."
        
        # Build history prompt
        history_prompt = ""
        for msg in history:
            history_prompt += f"{msg['role']}: {msg['content']}\n"
        
        # Join contexts
        context = "\n\n".join(contexts)
        
        # Detect formatting requirements
        query_lower = query.lower()
        formatting_instructions = ""
        
        if any(word in query_lower for word in ['points', 'list', 'bullet', 'list down']):
            formatting_instructions = """
FORMATTING REQUIREMENT: The user wants bullet points.
Structure your answer as:
• Point 1
• Point 2
• Point 3
Use clear, concise bullet points."""
        
        elif any(word in query_lower for word in ['table', 'tabular', 'format as table', 'in table format']):
            formatting_instructions = """
CRITICAL: User wants table format. 

Create a simple table with pipes separating columns:

Column1 | Column2 | Column3
Value1 | Value2 | Value3
Value4 | Value5 | Value6

Keep it simple - just use | between columns. The frontend will render it beautifully."""

        # Enhanced system prompt with stronger table instructions
        system_instructions = """You are a healthcare analyst and clinical knowledge assistant.

CORE RESPONSIBILITIES:
- Answer questions based on the provided context
- Be accurate and cite specific data when available
- Present information clearly and professionally

CRITICAL FORMATTING RULES:
1. BULLET POINTS: When user asks for "points" or "list", use this format:
   • First point
   • Second point
   • Third point

2. TABLES: When user asks for "table format", create a clean ASCII table:
   
   Column 1        | Column 2  | Column 3
   ----------------|-----------|----------
   Data A          | Value 1   | Info X
   Data B          | Value 2   | Info Y
   
   IMPORTANT: Use pipe (|) separators, align columns, use dashes for headers

3. PARAGRAPHS: For regular questions, use clear paragraph format

ANSWER GUIDELINES:
- If direct answer exists in context, provide it clearly
- If partial information exists, summarize relevant details
- If no information is found, state: "I do not have enough information in the provided documents."
- ALWAYS follow the user's requested format exactly
- For tables: ensure proper alignment and spacing"""

        prompt = f"""{system_instructions}

{formatting_instructions}

Conversation so far:
{history_prompt}

Context:
{context}

Question: {query}

IMPORTANT INSTRUCTIONS:
- Extract data from context and present it in the requested format
- If table is requested, create a properly formatted ASCII table with | separators
- Align columns and use consistent spacing
- Do not create broken or malformed tables

Your answer:"""

        for attempt in range(1, retries + 1):
            try:
                logging.info(f"LLM request attempt {attempt}")

                response = self.client.generate_completion(
                    prompt=prompt,
                    temperature=0.2,
                    max_tokens=1000,
                )

                answer = response["choices"][0]["message"]["content"]
                
                # No need for post-processing - frontend handles table rendering
                return answer

            except requests.exceptions.ConnectionError as e:
                logging.warning(f"LLM connection error (attempt {attempt}): {e}")
                time.sleep(0.7 * attempt)

            except Exception as e:
                logging.error(f"LLM unexpected error: {e}", exc_info=True)
                break

        logging.error("LLM generation failed after retries")
        return "Sorry, I'm having trouble generating a response right now."
    
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