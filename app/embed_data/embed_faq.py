import os
import shutil
import json
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path

# Paksa hanya gunakan CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# === Konfigurasi ===
JSON_PATH = "app/faq.json"
PERSIST_DIR = "db_faq_baru_3"

# === Hapus direktori lama jika ada ===
if os.path.exists(PERSIST_DIR):
    print(f"🧹 Menghapus direktori lama: {PERSIST_DIR}")
    shutil.rmtree(PERSIST_DIR)

# === Load data JSON ===
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# === Konversi ke Dokumen LangChain ===
docs = []
for i, item in enumerate(data):
    question = item.get("question", "").strip()
    answer = item.get("answer", "").strip()
    topic = item.get("topic", "").strip()

    if not question or not answer:
        print(f"⚠️  Lewatkan data index {i} karena kosong.")
        continue

    docs.append(
        Document(
            page_content=answer,
            metadata={
                "question": question,
                "topic": topic,
                "id": str(i)
            }
        )
    )

print(f"✅ Total dokumen valid: {len(docs)}")

# === Inisialisasi Embedding ===
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# === Simpan ke Chroma ===
db = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=PERSIST_DIR
)

print("✅ Embedding berhasil disimpan ke Chroma (CPU).")
