import os

from dotenv import load_dotenv 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_groq import ChatGroq

from src.format_docs import format_docs
from src.vectorstore import get_vectorstore

load_dotenv()

vectorstore = get_vectorstore()
retriever1 = vectorstore.as_retriever()

model = ChatGroq(model="openai/gpt-oss-120b")



prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant answering questions about company annual reports. "
            "Answer the question using only the context provided below. "
            "If the context doesn't contain the answer, say you don't know — do not make anything up.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def build_rag_chain(retriever):
    return(
        RunnableParallel(
            context = retriever | format_docs,
            question = RunnablePassthrough()
        )
        | prompt | model | StrOutputParser()
    )

chain1 = build_rag_chain(retriever1)


if __name__ =="__main__":
    question = "what is Nvidia's total revenue of the fiscal year 2025"
    answer = chain1.invoke(question)
    print(answer)