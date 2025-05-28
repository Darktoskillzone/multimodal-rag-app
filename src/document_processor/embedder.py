from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def embed_text(texts: list) -> list:
    """
    Function to embed a list of text chunks using the BGE model.
    
    Args:
        texts (list): List of text chunks to be embedded.
        
    Returns:
        list: List of embeddings corresponding to the input text chunks.
    """
    # Get embeddings
    embeddings = model.encode(texts, normalize_embeddings=True)  # important for cosine similarity
    return embeddings

if __name__ == "__main__":
    # Your text chunks (from chunking step)
    texts = [
        "What is the capital of France?",
        "Paris is the capital and most populous city of France.",
        "Berlin is the capital of Germany."
    ]

    embeddings = embed_text(texts)
    for i, embedding in enumerate(embeddings):
        print(f"Embedding {i+1}: {embedding[:5]}...")  # Print first 5 dimensions of the embedding
        print(f"Embedding {i+1} length: {len(embedding)} dimensions")