from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
import shutil
from langchain_chroma import Chroma

from src.ingest import load_and_split

embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 

collection_name  = "annual_reports"

VECTORSTORE_DIR = Path(__file__).resolve().parent.parent / "chromadb"

def build_vectorstore(force: bool = False):
    # Check whether the database already exists
    if(
        not force
        and VECTORSTORE_DIR.exists()
        and any(VECTORSTORE_DIR.iterdir())
    ):
        print("loading existing chroma db...")

        return Chroma(
            collection_name = collection_name,
            embedding_function = embedding_model,
            persist_directory = str(VECTORSTORE_DIR)
        )

    if force and VECTORSTORE_DIR.exists():
        print(f"Force rebuild: removing existing vector store directory at {VECTORSTORE_DIR}")
        shutil.rmtree(VECTORSTORE_DIR)

    print("Creating new Vector db")

    chunks  = load_and_split()

    vectorstore = Chroma.from_documents(
        documents= chunks,
        embedding = embedding_model,
        collection_name= collection_name,
        persist_directory= str(VECTORSTORE_DIR)
    )

    return vectorstore

def get_vectorstore():
    return Chroma(
        collection_name= collection_name,
        embedding_function= embedding_model,
        persist_directory= str(VECTORSTORE_DIR)
    )

if __name__ == "__main__":
    vectorstore = build_vectorstore(force=False)
    print(f"Vectorstore created with {vectorstore._collection.count()} documents")
