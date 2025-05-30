import os

from backend.embedder import Embedder
from backend.processor import Processor

def process_document(input_file_path: str):
    embedder = Embedder("BAAI/bge-base-en-v1.5")
    processor = Processor()
    # Example usage of ingesting a PDF and embedding its content
    documents = processor.ingest(input_file_path)
    
    for doc in documents:
        # Embed the text content of the document
        embedding = embedder.embed(doc.page_content)  # Get the first embedding
        processor.send_to_db(doc.page_content, embedding.tolist())

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "samples", "sample.pdf")
    process_document(file_path)
    print("Inserted document with embedding.")