from src.chain import retriever1,model
from operator import itemgetter
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from src.chain import chain1
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from src.format_docs import format_docs
from langchain_core.runnables import RunnableParallel
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


prompt  = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant answering questions about the company's annual report."
    "Use the context and conversation history to answer. Context:{context}"),
    MessagesPlaceholder(variable_name='history'),
    ("human", "{question}")
])

chain = (RunnableParallel(
    context = itemgetter("question") |
    retriever1 | format_docs ,
    question = itemgetter("question"),
    history = itemgetter("history")
)
    | prompt | model | StrOutputParser()
)

session_store = {}

def session_store_get(session_id:str):
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    chain ,
    get_session_history = session_store_get ,
    input_messages_key="question",
    history_messages_key="history"
)


if __name__ == "__main__":
    config = {"configurable": {"session_id": "test-session-1"}}

    response_1 = chain_with_memory.invoke(
        {"question": "What was NVIDIA's Data Center revenue growth?"}, config
    )
    print(response_1+"\n")
    response_2 = chain_with_memory.invoke(
        {"question": "How does that compare to its Gaming segment?"}, config
    )
    print(response_2)

    