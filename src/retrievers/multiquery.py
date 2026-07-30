from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from src.format_docs import format_docs
from src.vectorstore import get_vectorstore
from langchain_core.prompts import ChatPromptTemplate
from src.chain import model
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from eval.questions import EVAL_QUESTION
# Wrapping basic retriever around the multiquery retriever
retriever = get_vectorstore().as_retriever()
multi_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=model)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an original question, generate a list of alternative questions that cover the same information from different perspectives."
            "Focus on varying terminology, scope, and specificity."
            "Return each alternative as a separate line."
            "context:{context}"
        ),
        (
            "human",
            "question:{question}"
        )
    ]
)

chain_multi = (
    RunnableParallel(
        context=(lambda x: x["question"]) | multi_retriever | format_docs,
        question=lambda x: x["question"]
    )
    | prompt | model | StrOutputParser()
)

if __name__ == "__main__":
    for q in EVAL_QUESTION:
        question = q["question"]
        print(question)
        result = multi_retriever.invoke(question)
        for r in result:
            print(r)
