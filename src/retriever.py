from embeddings import EmbeddingManager
from vectore_store import VectorStore


class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedder: EmbeddingManager):
        self.vector_store = vector_store
        self.embedder = embedder
        

    def retrieve(self, query: str, top_k: int = 3, score_threshold: float = 0.2):
   
        query_emb = self.embedder.embed_texts([query])[0]

        results = self.vector_store.search(query_emb.tolist(), k=top_k)
        # print("\nSCORES:")
        # for s, doc in zip(results["distances"][0], results["documents"][0]):
        #     print(s, "---", doc[:120]) ## use them to check the distance 


        documents = []
        for doc, meta, score in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
     
            similarity = 1 - score

            if similarity >= score_threshold:
                documents.append({
                    "content": doc,
                    "metadata": meta,
                    "similarity_score": similarity
                })

        return documents
    
    

    
