import os
from dotenv import load_dotenv
from pathlib import Path

import psycopg

from backend.embedder import Embedder

class Retriever:
    def __init__(self, embedding_model_name: str = "BAAI/bge-base-en-v1.5"):

        env_path = Path(__file__).resolve().parents[2] / ".env"
        load_dotenv(dotenv_path=env_path)

        self.db_name = os.getenv("DEV_DB_NAME")
        self.db_user = os.getenv("DEV_DB_USER")

        self.embedder = Embedder(embedding_model_name)
    
    def retrieve(self, query, k=5):
        query_embedding = self.embedder.embed([query])[0]  # Get the embedding for the query

        conn = psycopg.connect(
            dbname=self.db_name,
            user=self.db_user,
        )
        cur = conn.cursor()

        cur.execute(
            """
            SELECT id, content
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, k)
        )

        results = cur.fetchall()
        cur.close()
        conn.close()
        return results
    
if __name__ == "__main__":
    retriever = Retriever("BAAI/bge-base-en-v1.5")
    embedder = Embedder("BAAI/bge-base-en-v1.5")
    # Example query embedding (should be a list of floats)
    query = "acroread"

    query_embedding = embedder.embed([query])  # Get the embedding for the query

    print("Embedding type:", type(query_embedding))
    print("Length:", len(query_embedding))
    print("First 3 vals:", query_embedding[:3])

    top_docs = retriever.retrieve(query, k=5)
    
    for doc in top_docs:
        print(f"Document ID: {doc[0]}, Content: {doc[1]}")
        print("\n")