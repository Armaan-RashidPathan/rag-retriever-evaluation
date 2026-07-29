import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from src.vectorstore import get_vectorstore
from src.format_docs import format_docs

# Load environment variables from .env file
load_dotenv()

def run_sample_query(user_prompt: str) -> str:
    """
    Sends a prompt to the OpenAI Chat model and returns the response.
    Requires OPENAI_API_KEY environment variable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY not found in environment. Please check your .env file."
        
    # Initialize the LLama model
    model = ChatGroq(model="llama-3.1-8b-instant")
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "{input}")
    ])
    
    # Chain components together
    chain = prompt | model | StrOutputParser()
    
    # Invoke the chain
    return chain.invoke({"input": user_prompt})

if __name__ == "__main__":
    print("Project initialized successfully.")
    print("Run `pytest` to verify setup.")
