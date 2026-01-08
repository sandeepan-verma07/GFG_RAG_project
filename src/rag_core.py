from src.llm_gemma import GemmaLLM
from src.mem0_client import get_user_memories

gemma = GemmaLLM()


def rag_answer(question: str, context_chunks, user_id: str, recent_messages: list = None):
    """
    Takes a user question + retrieved Qdrant/Web chunks,
    fetches long-term memories from Mem0 for this user,
    adds recent chat history, extracts readable text, and calls the LLM.
    
    recent_messages: list of {"role": "user"/"assistant", "content": str} from session_state
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

    # fetch user long term memories from Mem0
    memories = []
    if any(x in question.lower() for x in ["my", "me", "i ", "remember"]):
       memories = get_user_memories(user_id, question, limit=5)

    memory_text = "\n".join(memories) if memories else ""

    # Recent chat history (short term memory)
    recent_history = ""
    if recent_messages:
        recent_history = "RECENT CHAT HISTORY (use for 'last topic', 'we discussed' questions in a session, it may be empty if session is refreshed):\n"
        for msg in recent_messages:   #[-6:]:  # Last 3 turns (6 messages)
            role = "You: " if msg["role"] == "user" else "Assistant: "
            recent_history += f"{role}{msg['content']}\n"
        recent_history += "\n"

    return gemma.generate(question, context_text, memory_text=memory_text, recent_history=recent_history)
