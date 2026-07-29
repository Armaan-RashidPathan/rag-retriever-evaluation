import os
from src.main import run_sample_query

def test_run_sample_query_without_api_key(monkeypatch):
    """
    Verify that the sample query returns an error message when OPENAI_API_KEY is not set.
    """
    # Temporarily remove OPENAI_API_KEY if it exists
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    
    response = run_sample_query("Hello")
    assert "OPENAI_API_KEY not found" in response
