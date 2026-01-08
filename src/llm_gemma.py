# # src/llm_gemma.py
# from google import genai
# from dotenv import load_dotenv
# load_dotenv()

# import os


# class GemmaLLM:
#     def __init__(self, model_name="gemma-3-4b-it"):
#         api_key = os.getenv("GOOGLE_API_KEY")
#         if not api_key:
#             raise ValueError("GOOGLE_API_KEY not set in .env")

#         self.client = genai.Client(api_key=api_key)
#         self.model_name = model_name

#     def generate(self, question, context, memory_text: str = "", recent_history: str = ""):
#         """
#         `context` is your RAG context (PDF + web).
#         `memory_text` is long-term memories from Mem0.
#         `recent_history` is last few messages from current chat.
#         """

#         sections = []

#         if recent_history:
#             sections.append(recent_history)

#         if memory_text:
#             sections.append(f"""USER MEMORIES (use for personal/preference questions):
# {memory_text}""")

#         if context:
#             sections.append(f"""PDF/WEB CONTEXT (use for factual/document questions):
# {context}""")

#         full_context = "\n\n---\n\n".join(sections)

#         prompt = f"""
#         You are a helpful assistant. Answer using ONLY the provided context. 

#         STRICT RULES - NO EXCEPTIONS:
#         • NO EXTERNAL KNOWLEDGE. Ignore all "Famous Birthdays", "Wikipedia", or web results for personal info
#         • PERSONAL INFO ("my name", "I am") → ONLY USER MEMORIES or RECENT CHAT. Never guess/use web
#         • If NO personal info in memories/chat → say "I don't know your name yet. What's your name?"

#         NAME(or any critical information) HANDLING:
#         • "my name is X" → "Got it [X]! I'll remember that." (store in memory)
#         • "What's my name?" → Check memories/chat first, else "I don't know your name yet. Tell me!"

#         PRIORITY (check question first):
#         1. RECENT CHAT HISTORY → "last topic", "we discussed"
#         2. USER MEMORIES → personal facts/preferences  
#         3. PDF/WEB → documents only

#         {full_context}

#         Question: {question}

#         Answer naturally:
#         """

# src/llm_gemma.py
from google import genai
from dotenv import load_dotenv
load_dotenv()

import os


class GemmaLLM:
    def __init__(self, model_name="gemma-3-4b-it"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in .env")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, question, context, memory_text: str = "", recent_history: str = ""):
        sections = []

        if recent_history:
            sections.append(f"""RECENT CHAT HISTORY (scan ALL messages above):
{recent_history}""")

        if memory_text:
            sections.append(f"""LONG-TERM MEMORIES (scan EVERY line for personal info):
{memory_text}""")

        if context:
            sections.append(f"""PDF/WEB DOCUMENTS (facts only):
{context}""")

        full_context = "\n\n" + "="*80 + "\n\n".join(sections)

        prompt = f"""
       You are a RAG chatbot. You must answer the user strictly using the provided context.
Do NOT use your own prior knowledge.

========================
CONTEXT PRIORITY ORDER
========================

1. RECENT CHAT HISTORY  
   Use only for references like "we discussed earlier".

2. USER MEMORIES  
   Use only for personal information explicitly stored.

3. PDF CONTEXT  
   Use for document-based factual questions.

4. WEB SEARCH RESULTS  
   Use for general knowledge questions when PDFs or memories do not answer.

========================
STRICT PERSONAL INFO RULE
========================

• Questions about "my name", "who am I", "remember me" → SEARCH EVERY MEMORY LINE.
• If ANY memory says "User name is X" → use that name immediately.
• NEVER use web results for personal questions.

========================
WEB SEARCH HANDLING (CRITICAL)
========================

• If WEB SEARCH RESULTS are present, they are the PRIMARY source of truth.
• You MUST answer using WEB SEARCH RESULTS when they directly address the question.
• Do NOT replace web information with your own knowledge.
• Do NOT reject an answer just because it is approximate or not perfectly time-specific.
• Summarize the most relevant information from web results.

========================
WEB SEARCH RELEVANCE RULE
========================

• First, check if WEB SEARCH RESULTS DIRECTLY answer the question.
• "Directly answer" means the context clearly mentions the entity, place, or concept asked.
• If WEB SEARCH RESULTS are relevant → ALWAYS answer using them.
• If WEB SEARCH RESULTS exist but do NOT directly answer the question
  (e.g., vague summaries, unrelated topics),
  then treat this as NO relevant information.

========================
FAILSAFE RULE
========================

• ONLY if NO relevant information exists in:
  - RECENT CHAT HISTORY
  - USER MEMORIES
  - PDF CONTEXT
  - WEB SEARCH RESULTS

  respond exactly with:
  "I don't know yet. Can you clarify?"

========================
FACT VALIDATION RULE
========================

• If the question mentions a country, the answer MUST mention the SAME country.
• Ignore any context referring to a different country.

========================
TIME-SENSITIVE RULE
========================

• If the question asks for "current", "present", or "now":
• Use the MOST RECENT information available in WEB SEARCH RESULTS.
• Do NOT substitute with your own knowledge.
• If web information appears outdated or undated, respond:
  "Web sources may be outdated. Please confirm."

========================
OUTPUT RULE
========================

• Answer concisely.
• Do not mention sources.
• Do not explain reasoning.
• Do not hedge unless required by FAILSAFE RULE.

========================
CONTEXT
========================

{context}

========================
QUESTION
========================

{question}

        """



        response = self.client.models.generate_content(
            # model=self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        if hasattr(response, "output_text"):
            return response.output_text
        elif hasattr(response, "text"):
            return response.text
        else:
            return str(response)
