import pytest
from pathlib import Path
from langchain_core.documents import Document
from src.vectorstore import VECTORSTORE_DIR, build_vectorstore, get_vectorstore

def test_vectorstore_dir_is_correct():
    assert VECTORSTORE_DIR.name == "chromadb"
    assert VECTORSTORE_DIR.is_absolute()

def test_build_vectorstore_force_rebuild(monkeypatch):
    dummy_docs = [
        Document(page_content="Test document content 1", metadata={"source": "test"}),
        Document(page_content="Test document content 2", metadata={"source": "test"}),
    ]
    
    # Mock load_and_split to prevent it from scanning the real PDFs during testing
    monkeypatch.setattr("src.vectorstore.load_and_split", lambda: dummy_docs)
    
    # 1. Clean rebuild
    store = build_vectorstore(force=True)
    assert store._collection.count() == 2
    
    # 2. Loading without force loads existing data
    store_loaded = build_vectorstore(force=False)
    assert store_loaded._collection.count() == 2
