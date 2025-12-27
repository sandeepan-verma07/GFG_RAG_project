from src.vectore_store import VectorStore
from src.embeddings import EmbeddingManager
from src.retriever import RAGRetriever
from src.llm_gemma import GemmaLLM
from dotenv import load_dotenv
load_dotenv()


embedder = EmbeddingManager()
vector_store = VectorStore()

retriever = RAGRetriever(vector_store, embedder)
gemma = GemmaLLM()



def rag_answer(question: str):


    results = retriever.retrieve(question, top_k=3)

    if not results:
        return "No relevant context found."


    context = "\n\n".join([doc["content"] for doc in results])

    answer = gemma.generate(question, context)

    return answer
