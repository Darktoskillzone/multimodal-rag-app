from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, embedding_model_name: str = "BAAI/bge-base-en-v1.5"):
        """
        Initializes the Embedder class.
        
        Args:
            embedding_model_name (str): The name of the sentence transformer model to load.
                                         Defaults to "BAAI/bge-base-en-v1.5".
        
        This class is responsible for generating embedding vectors from text.
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)


    def embed(self, chunks: list) -> list:
        """
        Function to embed a list of text chunks using the BGE model.
        
        Args:
            texts (list): List of text chunks to be embedded.
            
        Returns:
            list: List of embeddings corresponding to the input text chunks.
        """
        # Get embeddings
        embeddings = self.embedding_model.encode(chunks, normalize_embeddings=True)  # important for cosine similarity
        return embeddings

if __name__ == "__main__":
    embedder = Embedder("BAAI/bge-base-en-v1.5")
    # Your text chunks (from chunking step)
    texts = [
        "What is the capital of France?",
        "Paris is the capital and most populous city of France.",
        "Berlin is the capital of Germany."
    ]

    embeddings = embedder.embed(texts)
    for i, embedding in enumerate(embeddings):
        print(f"Embedding {i+1}: {embedding[:5]}...")  # Print first 5 dimensions of the embedding
        print(f"Embedding {i+1} length: {len(embedding)} dimensions")