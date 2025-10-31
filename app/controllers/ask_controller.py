from app.service.rag_service import retriever, vector_db, reranker
from app.prompts.templates import rephrase_chain, chat_chain
from app.utils.ask_counter import increment_question_counter
from app.utils.ask_counter import load_counter


def rerank_by_metadata_question(question, documents,jumlah_data, top_k=30):
    pairs = []
    valid_docs = []

    # Siapkan pasangan pertanyaan dan metadata["question"]
    for doc in documents:
        if jumlah_data >1:
            meta_question = doc.metadata.get("question", "").strip()
        else :
            meta_question = doc.metadata.get("keywords", "").strip()
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

def rerank_documents(question, documents, top_k=1):
    pairs = [[question, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in reranked[:top_k]]

def process_question(question: str):
    jumlah_kata = len(question.strip().split())
    if "jumlah data" in question.lower():
        total_docs = vector_db._collection.count()
        return {
            "pertanyaan": question,
            "jawaban": f"Jumlah data RAG saya saat ini adalah {total_docs} dokumen."
        }

    retrieved_docs = retriever.invoke(question)
    if not retrieved_docs or all(not doc.page_content.strip() for doc in retrieved_docs):
        return {
            "pertanyaan": question,
            "jawaban": "Maaf, saya belum menemukan jawaban untuk pertanyaan tersebut."
        }

    reranked_meta = rerank_by_metadata_question(question, retrieved_docs,jumlah_kata)
    if not reranked_meta or reranked_meta[0][0] < 0:
        top_docs = rerank_documents(question, retrieved_docs)
        context = "\n\n".join([doc.page_content for doc in top_docs])
        answer = chat_chain.invoke({"question": question, "context": context})
        return {
            "pertanyaan": question,
            "pertanyaan_terkait": '',
            "jawaban": answer.content,
            "dokumen_terkait": ["Fallback ke LLM karena skor rendah"]
        }
    
    related_questions = []
    for score, doc in reranked_meta[1:20]:  # Skip top-1 (utama)
        print(score)
        if score > 0:
            question_meta = doc.metadata.get("question", "") or doc.metadata.get("keywords", "")
            jawaban = doc.page_content.strip()
            keywords = doc.metadata.get("keywords", "")

            if question_meta and jawaban:
                related_questions.append({
                    "pertanyaan": question_meta.strip(),
                    "jawaban": jawaban,
                    "keywords": keywords,
                    "score": float(score)
                })

    top_score, top_doc = reranked_meta[0]
    # Tambah log berapa kali pertanyaan metadata ini digunakan
    meta_q = top_doc.metadata.get("question", "").strip()
    if meta_q:
        increment_question_counter(meta_q)
    counter_data = load_counter()
    jumlah_ditanyakan = counter_data.get(meta_q, 1) 

    return {
        "pertanyaan": question,
        "jumlah_ditanyakan": jumlah_ditanyakan,
        "pertanyaan_terkait": meta_q,
        "keywords": top_doc.metadata.get("keywords", "").strip(),
        "jawaban": top_doc.page_content,
        "dokumen_terkait": [top_doc.page_content],
        "pertanyaan_terkait_lain": related_questions
        # "pertanyaan_terkait":[reranked_meta]
    }
