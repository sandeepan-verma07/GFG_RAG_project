import chromadb
from chromadb.utils import embedding_functions


class VectorStore:
    def __init__(self, persist_dir="vector_store"):
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Create (or get) collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"} 
        )

    def add_documents(self, texts, embeddings, metadatas=None, ids=None):
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def query(self, query_embedding, top_k=3):
        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
    def search(self, query_embedding, k=3):
        return self.collection.query(
             query_embeddings=[query_embedding],
             n_results=k
    )
    
