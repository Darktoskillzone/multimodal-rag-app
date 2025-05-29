import os
import dotenv
from pathlib import Path

import psycopg

from .embedder import embed_text

env_path = Path(__file__).resolve().parents[2] / ".env"
dotenv.load_dotenv(dotenv_path=env_path)

def get_top_k_similar_docs(query_embedding, k=5):
    conn = psycopg.connect(
        dbname=os.getenv("DEV_DB_NAME"),
        user=os.getenv("DEV_DB_USER"),
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
    # Example query embedding (should be a list of floats)
    query_embedding = embed_text("acroread").tolist()

    print("Embedding type:", type(query_embedding))
    print("Length:", len(query_embedding))
    print("First 3 vals:", query_embedding[:3])

    top_docs = get_top_k_similar_docs(query_embedding, k=5)
    
    for doc in top_docs:
        print(f"Document ID: {doc[0]}, Content: {doc[1]}")
        print("\n")