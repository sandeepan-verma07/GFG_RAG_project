from src.loader import load_and_chunk_pdfs
from src.embeddings import EmbeddingManager
from src.vectore_store import VectorStore

def run_ingest():
    pdf_folder = "./data"

    print(" Loading and chunking PDFs...")
    chunks = load_and_chunk_pdfs(pdf_folder)

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    embedder = EmbeddingManager()
    vector_store = VectorStore()

    print(" Creating embeddings...")
    embeddings = embedder.embed_texts(texts)

    print(" Storing vectors...")
    vector_store.add_documents(texts, embeddings, metadatas)

    print("Ingest completed")

if __name__ == "__main__":
    run_ingest()
