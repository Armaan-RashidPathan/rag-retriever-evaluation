from src.chain import model, build_rag_chain, retriever1
from src.pydantic_schemas import FiscalYearFinancials
from langchain_core.prompts import ChatPromptTemplate
from src.format_docs import format_docs
from langchain_core.runnables import RunnableParallel,RunnablePassthrough

structured_model = model.with_structured_output(FiscalYearFinancials, method="function_calling")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Extract financial data for the {query} year from the context:\n{context}"),
    ]
)

extraction_chain = prompt | structured_model


extraction_chain = (
    RunnableParallel(context=retriever1 | format_docs, query = RunnablePassthrough())
    | prompt
    | structured_model
)

if __name__ == "__main__":
    result = extraction_chain.invoke("NVIDIA fiscal year 2025 total revenue and segment breakdown")

    print(type(result))
    print(result)
    print("\nFirst segment:")
    print(result.segment_revenues[0])