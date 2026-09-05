from src.chain import model
from src.vectorstore import get_vectorstore
from src.pydantic_schemas import FiscalYearFinancials
from langchain_core.prompts import ChatPromptTemplate
from src.format_docs import format_docs
from langchain_core.runnables import RunnableParallel,RunnablePassthrough

# Explicit k=12 here (not the bare retriever1 from src.chain, which has no k
# set): same fix as Milestone 2's Contextual Compression finding — the target
# chunk with real FY2025 figures narrowly misses the default cutoff (~k=4).
retriever = get_vectorstore().as_retriever(search_kwargs={"k": 20})

structured_model = model.with_structured_output(FiscalYearFinancials, method="function_calling")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Extract financial data for the {query} year from the context:\n{context}"),
    ]
)

extraction_chain = (
    RunnableParallel(context=retriever | format_docs, query = RunnablePassthrough())
    | prompt
    | structured_model
)

if __name__ == "__main__":
    result = extraction_chain.invoke("NVIDIA fiscal year 2025 total revenue and segment breakdown")

    print(type(result))
    print(result)
    print("\nFirst segment:")
    print(result.segment_revenues[0])