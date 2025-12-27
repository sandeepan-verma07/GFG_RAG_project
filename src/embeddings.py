from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np


class EmbeddingManager:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = HuggingFaceEmbeddings(model_name=model_name)

    def embed_texts(self, texts):
        embeddings = self.model.embed_documents(texts)
        return np.array(embeddings)
    