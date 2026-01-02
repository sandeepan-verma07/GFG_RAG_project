from src.llm_gemma import GemmaLLM

gemma = GemmaLLM()

def rag_answer(question: str, context_chunks):
    """
    Takes a user question + retrieved Qdrant/Web chunks,
    extracts readable text, and calls the LLM.
    """

    def extract_text(chunk):
        # Web + Qdrant formats
        if isinstance(chunk, dict):
            if "text" in chunk:
                return str(chunk["text"])

            if "content" in chunk:
                return str(chunk["content"])

            # Some Qdrant payload styles
            if "payload" in chunk and "text" in chunk["payload"]:
                return str(chunk["payload"]["text"])

        # Fallback — force string
        return str(chunk)

    if not context_chunks:
        context_text = ""
    else:
        context_text = "\n\n".join(
            extract_text(c) for c in context_chunks
        )

    return gemma.generate(question, context_text)
