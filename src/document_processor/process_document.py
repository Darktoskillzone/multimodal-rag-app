import psycopg

from ingester import ingest_pdf
from embedder import embed_text

def insert_document(content, embedding):
    with psycopg.connect("dbname=vector_db user=sheep") as conn:
        with conn.cursor() as cur:
            # The embedding vector is passed as a Python list,
            # psycopg will convert it to PostgreSQL vector automatically.
            cur.execute(
                "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
                (content, embedding)
            )
        conn.commit()

def process_document(input_file_path: str):
    # Example usage of ingesting a PDF and embedding its content
    documents = ingest_pdf(input_file_path)
    
    for doc in documents:
        # Embed the text content of the document
        embedding = embed_text(doc.page_content)  # Get the first embedding
        insert_document(doc.page_content, embedding.tolist())

if __name__ == "__main__":
    process_document("samples/sample.pdf")
    print("Inserted document with embedding.")