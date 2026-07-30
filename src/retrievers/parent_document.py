import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from eval.questions import EVAL_QUESTION
from src.chain import build_rag_chain
from src.ingest import load_documents

load_dotenv()

PARENT_CHILD_DIR = Path(__file__).resolve().parent.parent.parent / "chromadb_parent_child"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Splits used to build the parent index: small chunks for search, big chunks for context.
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)


def build_parent_document_retriever() -> ParentDocumentRetriever:
    # InMemoryStore never persists across runs, so there's no safe way to reuse an
    # old persisted vectorstore without the docstore falling out of sync with it.
    # Simplest correct approach: always wipe and rebuild both together.
    if PARENT_CHILD_DIR.exists():
        shutil.rmtree(PARENT_CHILD_DIR)

    vectorstore = Chroma(
        collection_name="parent_child_children",
        embedding_function=embeddings,
        persist_directory=str(PARENT_CHILD_DIR),
    )
    docstore = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 12},
    )

    raw_docs = load_documents()

    # Chroma caps how many items can be upserted in a single call. Splitting ~530
    # pages into 400-char child chunks produces thousands of chunks at once, so we
    # add the raw (page-level) documents in smaller batches instead of all at once.
    batch_size = 50
    for i in range(0, len(raw_docs), batch_size):
        batch = raw_docs[i : i + batch_size]
        retriever.add_documents(batch)
        print(f"  added batch {i // batch_size + 1} ({len(batch)} pages)")

    return retriever


if __name__ == "__main__":
    print("Building parent document retriever (re-embeds everything, takes a few minutes)...")
    parent_retriever = build_parent_document_retriever()

    question = EVAL_QUESTION[0]["question"]

    print("=" * 80)
    print("QUESTION")
    print(question)

    docs = parent_retriever.invoke(question)
    print(f"\nRetrieved {len(docs)} parent document(s)\n")
    for i, doc in enumerate(docs, start=1):
        print(f"Document {i} ({len(doc.page_content)} chars)")
        print("-" * 80)
        print(doc.page_content)
        print()

    chain = build_rag_chain(parent_retriever)
    answer = chain.invoke(question)

    print("=" * 80)
    print("ANSWER (via Parent Document Retriever)")
    print(answer)
