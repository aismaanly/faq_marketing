from fastapi import FastAPI
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from sentence_transformers import CrossEncoder

# === Inisialisasi FastAPI ===
app = FastAPI(title="Roxy AI - TEST Gold Assistant")

# === Load Embedding Model ===
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# === Load Vector Database ===
vector_db = Chroma(
    persist_directory="db_faq_baru_3",
    embedding_function=embedding_model
)
retriever = vector_db.as_retriever(search_kwargs={"k": 10})

# === Load CrossEncoder untuk Reranker ===
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device='cpu')

# === Load Chat Model dari Ollama ===
llm = ChatOllama(
    model="roxy-ai",
    temperature=0.0,
)

# === Prompt Template ===
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "Kamu adalah Roxy AI, asisten virtual resmi dari TEST Gold. "
     "Diciptakan oleh tim ICT (Information Communication Technology) untuk membantu pengguna TEST Gold. "
     "Jawaban kamu harus profesional, jelas, akurat, dan selalu dalam bahasa Indonesia. "
     "Jawaban HANYA boleh berdasarkan data yang diberikan dari RAG (`db_faq_baru_3`). "
     "Jika tidak menemukan jawabannya di data RAG, katakan 'Maaf, saya tidak menemukan informasi tersebut dalam data kami.' "
     "ULANGI pencarian dalam RAG secara menyeluruh hingga 100 kali harus ketemu, terutama jika ditanya tentang nama personal. "
     "JANGAN berimajinasi atau membuat informasi tambahan yang tidak ada dalam data RAG."),
    ("human", 
     "Berikut adalah data hasil pencarian:\n\n{context}\n\n"
     "Pertanyaan: {question}\n\n")
])

# === Build LLM Chain (tanpa deprecated LLMChain) ===
chain: RunnableSequence = chat_prompt | llm

# === Reranking Function ===
def rerank_documents(question, documents, top_k=1):
    # Kita bandingkan dengan metadata pertanyaannya
    pairs = [[question, doc.metadata.get("question", "")] for doc in documents]
    scores = reranker.predict(pairs)
    
    # Urutkan berdasarkan skor
    reranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    
    # Kembalikan hanya dokumen terbaik
    return [doc for _, doc in reranked[:top_k]]


# === Skema Request ===
class QuestionRequest(BaseModel):
    question: str

# === Endpoint untuk Bertanya ===
@app.post("/ask")
def ask_roxy(request: QuestionRequest):
    question = request.question

    # Khusus pertanyaan "jumlah data"
    if "jumlah data" in question.lower():
        total_docs = vector_db._collection.count()
        return {
            "pertanyaan": question,
            "jawaban": f"Jumlah data RAG saya saat ini adalah {total_docs} dokumen."
        }

    # Ambil dokumen
    retrieved_docs = retriever.get_relevant_documents(question)
    if not retrieved_docs or all(len(doc.page_content.strip()) == 0 for doc in retrieved_docs):
        return {
            "pertanyaan": question,
            "jawaban": "Maaf, saya belum menemukan jawaban untuk pertanyaan tersebut."
        }

    # Rerank dokumen
    top_docs = rerank_documents(question, retrieved_docs, top_k=1)
    doc_texts = [doc.page_content for doc in top_docs]
    context = "\n\n".join(doc_texts)

    # Jalankan pipeline prompt | llm
    answer = chain.invoke({"question": question, "context": context})

    return {
        "pertanyaan": question,
        "jawaban": answer,
        "dokumen_terkait": doc_texts
    }

# === Root Endpoint ===
@app.get("/")
def root():
    return {"message": "Roxy AI API with ChatOllama is running"}
