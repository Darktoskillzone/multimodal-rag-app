import pymupdf4llm
from langchain.text_splitter import RecursiveCharacterTextSplitter

def ingest_pdf(input_file_path: str) -> list:
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

if __name__ == "__main__":
    documents = ingest_pdf("samples/sample.pdf")
    for i, doc in enumerate(documents):
        print(f"\n--- Paragraph {i+1} ---")
        print(doc.page_content)
        print(f"Length: {len(doc.page_content)} characters")