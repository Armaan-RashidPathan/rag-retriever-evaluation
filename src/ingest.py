from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    docs = []
    for pdf_path in data_dir.glob("*.pdf"):
        loader = PyPDFLoader(file_path=str(pdf_path))
        docs.extend(loader.load())
    return docs


def load_and_split(
    data_dir: Path = DATA_DIR,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    docs = load_documents(data_dir)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


if __name__ == "__main__":
    chunks = load_and_split()

    print(f"Total chunks: {len(chunks)}")
    if chunks:
        print("\nMetadata of first chunk:")
        print(chunks[0].metadata)
        print("\nPreview of first chunk:")
        print(chunks[0].page_content[:500])
    else:
        print("No documents were loaded. Check the data directory and PDF files.")