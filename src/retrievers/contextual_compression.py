from dotenv import load_dotenv
from src.vectorstore import get_vectorstore
from src.chain import model, build_rag_chain
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
from langchain_huggingface import HuggingFaceEmbeddings
from eval.questions import EVAL_QUESTION

load_dotenv()

#base retriever

vectorstore = get_vectorstore()

base_retriever = vectorstore.as_retriever(
    search_kwargs={"k":12}
)

#Same embedding mmodel used to build the vector store

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#filtering documents based on similarity score
embedding_filter = EmbeddingsFilter(
    embeddings = embeddings,
    similarity_threshold = 0.65
)

# Wrap the base retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=embedding_filter,
    base_retriever=base_retriever,
)

chain = build_rag_chain(compression_retriever)

if __name__ == "__main__":
    question = EVAL_QUESTION[0]["question"]

    answer = chain.invoke(question)

    print("=" * 80)
    print("QUESTION")
    print(question)
    print("\n" + "=" * 80)
    print("ANSWER (via Contextual Compression Retriever)")
    print(answer)