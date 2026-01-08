from mem0 import MemoryClient
from dotenv import load_dotenv
import os

load_dotenv()

mem0_client = MemoryClient(
    api_key=os.getenv("MEM0_API_KEY"),
)


def add_user_memories(user_id: str, messages: list):
    """
    Add conversation messages to Mem0 as long-term memory for a given user.

    messages: list of {"role": "user"/"assistant", "content": str}
    """
    if not messages:
        return
    try:
        mem0_client.add(
            messages=messages, 
            user_id=user_id  # scopes to the active user id
        )
        print(f"Added memories for user {user_id}")
    except Exception as e:
        print(f"Mem0 add failed: {e}")


def get_user_memories(user_id: str, query: str, limit: int = 5) -> list:
    """
    Retrieve relevant memories for this user and query from Mem0.

    Uses filters={"user_id": user_id} to scope correctly.
    """
    try:
        results = mem0_client.search(
            query=query,
            filters={"user_id": user_id},                       # for multitenancy
            limit=limit
        )
    except Exception as e:
        print(f"Mem0 search failed: {e}")
        return []

    mem_texts = []
    if results and isinstance(results, dict) and "results" in results:
        for m in results["results"]:
            if isinstance(m, dict) and "memory" in m:
                mem_texts.append(m["memory"])
    elif isinstance(results, list):
        for m in results:
            if isinstance(m, dict) and "memory" in m:
                mem_texts.append(m["memory"])

    

    print(mem_texts)

    return mem_texts
