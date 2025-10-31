from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

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
    persist_directory="db_faq_baru_4",
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

# prompt = ChatPromptTemplate.from_template("""
# Berikut adalah jawaban mentah yang ditemukan:
# "{answer_raw}"

# Tolong tulis ulang jawaban ini agar lebih jelas dan brutal, serta mudah dipahami oleh pelanggan awam.
# jawab menggunakan bahasa yang lugas tanpa menjelaskan apapun selain dari pengembangan data tersebut.
# """)
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "Tolong tulis ulang jawaban ini agar lebih jelas dan brutal dan kejam ketika membahas personal, serta mudah dipahami oleh pelanggan awam."
     "jawab menggunakan bahasa yang lugas tanpa menjelaskan apapun selain dari pengembangan data tersebut pakailah bahasa indonesia."),
    ("human", 
     "Berikut adalah jawaban mentah yang ditemukan:\n\n{answer_raw}\n\n")
])
rephrase_chain = prompt | llm
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

# === Skema Request ===
class QuestionRequest(BaseModel):
    question: str

# === Fungsi Rerank Metadata ===
def rerank_documents(question, documents, top_k=1, return_scores=False):
    pairs = [[question, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

    if return_scores:
        return reranked[:top_k]  # [(score, doc), ...]
    return [doc for _, doc in reranked[:top_k]]

def rerank_by_metadata_question(question, documents, top_k=1):
    pairs = []
    valid_docs = []

    # Siapkan pasangan pertanyaan dan metadata["question"]
    for doc in documents:
        meta_question = doc.metadata.get("question", "").strip()
        if meta_question:
            pairs.append([question, meta_question])
            valid_docs.append(doc)

    if not pairs:
        print("⚠️ Tidak ada metadata 'question' yang tersedia untuk reranking.")
        return []

    # Prediksi skor menggunakan reranker
    scores = reranker.predict(pairs)

    # Cetak debug info
    print(f"\n🔍 Reranking berdasarkan metadata 'question':")
    for i, (score, doc, pair) in enumerate(zip(scores, valid_docs, pairs)):
        print(f"[{i}] Score: {score:.3f} | UserQ: \"{pair[0]}\" | MetaQ: \"{pair[1]}\"")

    # Urutkan berdasarkan skor tertinggi
    reranked = sorted(zip(scores, valid_docs), key=lambda x: x[0], reverse=True)

    return reranked[:top_k]



# === Endpoint Bertanya ===
@app.post("/ask")
def ask_roxy(request: QuestionRequest):
    question = request.question.strip()

    # === Cek pertanyaan khusus ===
    if "jumlah data" in question.lower():
        total_docs = vector_db._collection.count()
        return {
            "pertanyaan": question,
            "jawaban": f"Jumlah data RAG saya saat ini adalah {total_docs} dokumen."
        }

    # === Ambil dokumen ===
    retrieved_docs = retriever.invoke(question)  # gunakan .invoke() sesuai LangChain v0.1.46+
    if not retrieved_docs or all(not doc.page_content.strip() for doc in retrieved_docs):
        return {
            "pertanyaan": question,
            "jawaban": "Maaf, saya belum menemukan jawaban untuk pertanyaan tersebut."
        }

    # === Rerank berdasarkan metadata["question"] ===
    reranked_meta = rerank_by_metadata_question(question, retrieved_docs, top_k=1)

    if not reranked_meta:
        fallback_reason = "⚠️ Tidak ada metadata 'question' tersedia, fallback ke Ollama."
        top_doc = None
        top_score = -999
    else:
        top_score, top_doc = reranked_meta[0]
        fallback_reason = f"⚠️ Score terlalu rendah ({top_score:.3f}), fallback ke Ollama." if top_score < 0 else None

    # === Jika fallback ke LLM Ollama ===
    if top_doc is None or top_score < 0:
        # Ambil dokumen berdasarkan page_content lalu olah dengan LLM
        top_docs = rerank_documents(question, retrieved_docs, top_k=1)
        context = "\n\n".join([doc.page_content for doc in top_docs])
        answer = chain.invoke({"question": question, "context": context})

        return {
            "pertanyaan": question,
            "jawaban": answer.content,
            "dokumen_terkait": [fallback_reason]
        }

    # === Jika skor OK, rephrase jawaban dari dokumen ===
    answer_rephrased = rephrase_chain.invoke({"answer_raw": top_doc.page_content})

    return {
        "pertanyaan": question,
        "jawaban": answer_rephrased.content,
        "dokumen_terkait": [top_doc.page_content]
    }

@app.get("/data")
def get_all_data():
    try:
        # Ambil semua dokumen dari koleksi
        all_docs = vector_db._collection.get(include=["metadatas", "documents"])
        
        # Format hasil sebagai list of dict
        result = []
        for doc_id, doc_content in zip(all_docs["ids"], all_docs["documents"]):
            metadata = all_docs["metadatas"][all_docs["ids"].index(doc_id)]
            result.append({
                "id": doc_id,
                "konten": doc_content,
                "metadata": metadata
            })

        return JSONResponse(content={"jumlah": len(result), "data": result}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# === Endpoint Root ===
@app.get("/")
def root():
    return {"message": "Roxy AI API (tanpa LLM) is running"}
