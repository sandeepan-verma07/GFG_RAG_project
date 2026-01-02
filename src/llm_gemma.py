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
    def generate(self, question, context):

        prompt = f"""
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is unclear, say "The document does not clearly answer this."
Context:
{context}

Question: {question}

 Answer in 3-5 sentences, clear and specific:
"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

     
        if hasattr(response, "output_text"):
            return response.output_text
        elif hasattr(response, "text"):
            return response.text
        else:
            return str(response)
