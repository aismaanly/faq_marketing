from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

# Embedding
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Vector DB
vector_db = Chroma(
    persist_directory="db_faq_baru_4",
    embedding_function=embedding_model
)
retriever = vector_db.as_retriever(search_kwargs={"k": 100})

# Reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device='cpu')

# LLM
llm = ChatOllama(model="roxy-ai", temperature=0.0)
