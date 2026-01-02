from fastembed import TextEmbedding
import numpy as np

def embed_texts(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = TextEmbedding(model_name=model_name)

    clean_texts = [
        t if isinstance(t, str)
        else t.get("text") if isinstance(t, dict)
        else getattr(t, "page_content", "")
        for t in texts
    ]

    embeddings_generator = model.embed(clean_texts)

    # Extract all embeddings from the generator
    embeddings = list(embeddings_generator)

    return embeddings

def embed_query(query, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = TextEmbedding(model_name=model_name)
    embedding_generator = model.embed([query])

    # Extract the embedding from the generator
    embedding = next(embedding_generator)

    return embedding



def get_embedding_size(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = TextEmbedding(model_name=model_name)

    embedding_generator = model.embed(["test"])   #it is a generator

    first_embedding = next(embedding_generator)

    vector_dimension = len(first_embedding)

    return vector_dimension

if __name__=="__main__":
    print("Vector Dimension:", get_embedding_size())