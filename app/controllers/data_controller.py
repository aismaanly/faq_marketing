from fastapi.responses import JSONResponse
from app.service.rag_service import vector_db
from app.utils.ask_counter import load_counter, load_json, save_json
from collections import defaultdict
from langchain_core.documents import Document

def get_all_data():
    try:
        all_docs = vector_db._collection.get(include=["metadatas", "documents"])
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

def get_faq_detail_by_id(id: str):
    try:
        # Ambil dokumen dari vector DB berdasarkan ID
        result = vector_db._collection.get(ids=[id], include=["metadatas", "documents"])

        if not result["ids"]:
            return None  # Tidak ditemukan

        return {
            "id": result["ids"][0],
            "konten": result["documents"][0],
            "metadata": result["metadatas"][0]
        }
    except Exception as e:
        raise Exception(f"Gagal mengambil FAQ ID {id}: {str(e)}")


def get_top_pertanyaan():
    try:
        counter_data = load_counter()

        # Ambil 10 pertanyaan terbanyak
        top_questions = sorted(counter_data.items(), key=lambda x: x[1], reverse=True)[:10]

        result = []
        for question_text, count in top_questions:
            # Cari dokumen yang punya metadata question yang cocok
            docs = vector_db._collection.get(
                where={"question": {"$eq": question_text}},
                include=["documents", "metadatas"]
            )
            if docs["documents"]:
                result.append({
                    "pertanyaan": question_text,
                    "jumlah_ditanyakan": count,
                    "keywords": docs["metadatas"][0].get("keywords", ""),
                    "jawaban": docs["documents"][0]  # page_content
                })

        return {
            "top_pertanyaan": result
        }

    except Exception as e:
        return {"error": str(e)}

def all_topic():
    try:
        all_docs = vector_db._collection.get(include=["metadatas", "documents"])
        grouped_data = defaultdict(list)

        for i, doc_id in enumerate(all_docs["ids"]):
            doc_content = all_docs["documents"][i]
            metadata = all_docs["metadatas"][i]

            topic = metadata.get("topic", "Unknown")  # gunakan "topik" jika pakai bahasa Indonesia
            grouped_data[topic].append({
                "id": doc_id,
                "konten": doc_content,
                "metadata": metadata
            })

        # format hasil jadi list of dict
        result = [{"topic": topic, "data": docs} for topic, docs in grouped_data.items()]

        return JSONResponse(content={
            "jumlah": len(result),
            "data": result
        }, status_code=200)


    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

def add_faq_to_chroma_and_json(item: dict) -> bool:
    try:
        # === 1. Load JSON lama ===
        data = load_json()
        new_id = str(len(data))  # ID sebagai string (untuk Chroma)

        # === 2. Siapkan format baru ===
        faq_item = {
            "pertanyaan": item.get("pertanyaan", "").strip(),
            "jawaban": item.get("jawaban", "").strip(),
            "kategori": item.get("kategori", "").strip(),
            "keywords": item.get("keywords", "").strip()
        }

        # Validasi minimal
        if not faq_item["pertanyaan"] or not faq_item["jawaban"]:
            return False

        # === 3. Simpan ke file JSON ===
        data.append(faq_item)
        save_json(data)

        # === 4. Simpan ke Vector DB (Chroma) ===
        doc = Document(
            page_content=faq_item["jawaban"],
            metadata={
                "id": new_id,
                "question": faq_item["pertanyaan"],
                "topic": faq_item["kategori"],
                "keywords": faq_item["keywords"]
            }
        )
        vector_db.add_documents([doc])
        # vector_db.persist()

        return True
    except Exception as e:
        print(f"❌ Gagal menambahkan FAQ: {e}")
        return False

def update_faq_by_id(id: str, item: dict) -> bool:
    try:
        # 1. Ambil data lama
        existing = vector_db._collection.get(ids=[id], include=["metadatas", "documents"])
        
        if not existing["ids"]:
            return False  # Tidak ditemukan

        # 2. Hapus dokumen lama dari vector DB
        vector_db._collection.delete(ids=[id])

        # 3. Buat dokumen baru
        updated_doc = Document(
            page_content=item.get("jawaban", "").strip(),
            metadata={
                "id": id,
                "question": item.get("pertanyaan", "").strip(),
                "topic": item.get("kategori", "").strip(),
                "keywords": item.get("keywords", "").strip()
            }
        )

        # 4. Tambahkan kembali ke vector DB
        vector_db.add_documents([updated_doc], ids=[id])

        # 5. (Opsional) Update file JSON
        data = load_json()
        for i, d in enumerate(data):
            if d.get("id") == id:
                data[i] = {
                    "id": id,
                    "question": item.get("pertanyaan", "").strip(),
                    "jawaban": item.get("jawaban", "").strip(),
                    "topic": item.get("kategori", "").strip(),
                    "keywords": item.get("keywords", "").strip()
                }
                break
        else:
            # Jika tidak ditemukan juga di file json, tambahkan baru
            data.append({
                "id": id,
                "pertanyaan": item.get("pertanyaan", "").strip(),
                "jawaban": item.get("jawaban", "").strip(),
                "kategori": item.get("kategori", "").strip(),
                "keywords": item.get("keywords", "").strip()
            })

        save_json(data)

        return True

    except Exception as e:
        print(f"❌ Gagal update FAQ ID {id}: {e}")
        return False

def delete_faq_by_id(id: str) -> bool:
    try:
        # 1. Cek apakah ID ada di vector DB
        existing = vector_db._collection.get(ids=[id], include=["metadatas"])
        if not existing["ids"]:
            return False  # ID tidak ditemukan

        # 2. Hapus dari vector DB
        vector_db._collection.delete(ids=[id])

        # 3. Hapus dari JSON
        data = load_json()
        new_data = [d for d in data if d.get("id") != id]
        if len(new_data) == len(data):
            # Tidak ada yang dihapus dari file JSON
            return False

        save_json(new_data)
        return True
    except Exception as e:
        print(f"❌ Gagal menghapus FAQ ID {id}: {e}")
        return False