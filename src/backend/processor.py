import os
import dotenv
from pathlib import Path

import psycopg

import pymupdf4llm
from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend.embedder import Embedder

env_path = Path(__file__).resolve().parents[2] / ".env"
dotenv.load_dotenv(dotenv_path=env_path)

# Ideally this will be a class that can be extended for different types of documents
class Processor:
    def __init__(self, embedding_model_name: str = "BAAI/bge-base-en-v1.5"):
        self.embedder = Embedder(embedding_model_name)
    def ingest(self, input_file_path: str) -> list:
        # Returns list of documents/chunks from unprocessed files
        """
        Function to process a PDF document and split it into manageable text chunks.
        This function uses pymupdf4llm to convert the PDF to markdown and then splits
        the text into smaller documents using RecursiveCharacterTextSplitter.
        """

        # Get the MD text
        md_text = pymupdf4llm.to_markdown(input_file_path)  # get markdown for all pages

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=0,
            separators=["\n\n", "\n", ".", ",", " "]
            )

        documents = splitter.create_documents([md_text])

        return documents
    
    def send_to_db(self, chunks: list):
        """s
        Function to generate chunk embeddings and send them to the database.
        """

        chunk_embeddings = self.embedder.embed(chunks)
    
        with psycopg.connect(
            dbname=os.getenv("DEV_DB_NAME"),
            user=os.getenv("DEV_DB_USER"),
        ) as conn:
            with conn.cursor() as cur:
                # The embedding vector is passed as a Python list,
                # psycopg will convert it to PostgreSQL vector automatically.
                cur.execute(
                    "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
                    (chunks, chunk_embeddings)
                )
            conn.commit()

if __name__ == "__main__":
    processor = Processor("BAAI/bge-base-en-v1.5")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    documents = processor.ingest(os.path.join(base_dir, "samples/sample.pdf"))
    # Print first 5 document chunks as examples
    for i, doc in enumerate(documents[:5]):
        print(f"\n--- Paragraph {i+1} ---")
        print(doc.page_content)
        print(f"Length: {len(doc.page_content)} characters")
