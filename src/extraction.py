from src.chain import model
from src.vectorstore import get_vectorstore
from src.pydantic_schemas import FiscalYearFinancials
from langchain_core.prompts import ChatPromptTemplate
from src.format_docs import format_docs
from langchain_core.runnables import RunnableParallel,RunnablePassthrough


from src.retrievers.parent_document import build_parent_document_retriever
# Explicit k=12 here (not the bare retriever1 from src.chain, which has no k
# set): same fix as Milestone 2's Contextual Compression finding — the target
# chunk with real FY2025 figures narrowly misses the default cutoff (~k=4).
retriever = get_vectorstore().as_retriever(search_kwargs={"k": 20})

# we faced issues when we were using the basic retriever(above one) because it is just a patch on our retrieval problem not the actual solution
# so we built a "parent retriever" 
# It uses two different chunk sizes at once, and the trick is in how they're used differently:
# Small "child" chunks (400 characters) — these are the ones that actually get embedded and searched against the question. Small chunks are good at precise matching, because they're not diluted with unrelated text.
# Large "parent" chunks (2000 characters) — these are what actually gets returned to you, whenever any child chunk inside them gets matched


parent_retriever = build_parent_document_retriever()

structured_model = model.with_structured_output(FiscalYearFinancials, method="function_calling")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human",
        """Extract the financial data requested in the query from the context.

        Query:
        {query}

        Context:
        {context}
        """),
    ]
)

extraction_chain = (
    RunnableParallel(context=parent_retriever| format_docs, query = RunnablePassthrough())
    | prompt
    | structured_model
)

if __name__ == "__main__":
    result = extraction_chain.invoke("NVIDIA fiscal year 2025 total revenue and segment breakdown")

    print(type(result))
    print(result)
    print("\nFirst segment:")
    print(result.segment_revenues[0])