from vectore_store import VectorStore
from embeddings import EmbeddingManager
from retriever import RAGRetriever
from llm_gemma import GemmaLLM
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
    
    print(results,"\n\n\n\n")


    context = "\n\n".join([doc["content"] for doc in results])
    print(context, "\n\n\n\n")

    answer = gemma.generate(question, context)

    return answer

print(rag_answer("What methodology is used in the research paper?"))