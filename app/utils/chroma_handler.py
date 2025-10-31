import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

JSON_PATH = "app/data/faq_baru.json"
PERSIST_DIR = "db_faq_baru_4"

# Load Chroma
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model
)

# Generate ID dari pertanyaan
def generate_id(pertanyaan: str) -> str:
    return "faq_" + str(abs(hash(pertanyaan)))

def add_faq_to_chroma_and_json(data: dict) -> str:
    try:
        id_ = generate_id(data["pertanyaan"])

        doc = Document(
            page_content=data["jawaban"],
            metadata={
                "question": data["pertanyaan"],
                "topic": data.get("kategori", ""),
                "keywords": data.get("keywords", ""),
                "id": id_
            }
        )

        db.add_documents([doc])

        # Tambahkan ke JSON
        if Path(JSON_PATH).exists():
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        else:
            existing = []

        existing.append(data)
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        return id_
    except Exception as e:
        print("❌ Error saat tambah:", e)
        return ""

def update_faq_by_id(id_: str, new_data: dict) -> bool:
    try:
        # Load data JSON
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)

        found = False
        for i, item in enumerate(existing):
            old_id = generate_id(item["pertanyaan"])
            if old_id == id_:
                existing[i] = new_data
                found = True
                break

        if not found:
            return False

        # Save JSON updated
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        # Hapus dokumen lama dan tambahkan dokumen baru
        db.delete([id_])
        doc = Document(
            page_content=new_data["jawaban"],
            metadata={
                "question": new_data["pertanyaan"],
                "topic": new_data.get("kategori", ""),
                "keywords": new_data.get("keywords", ""),
                "id": id_
            }
        )
        db.add_documents([doc])
        return True
    except Exception as e:
        print("❌ Error saat update:", e)
        return False

def delete_faq_by_id(id_: str) -> bool:
    try:
        # Load JSON
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)

        new_data = []
        found = False
        for item in existing:
            if generate_id(item["pertanyaan"]) == id_:
                found = True
                continue
            new_data.append(item)

        if not found:
            return False

        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)

        db.delete([id_])
        return True
    except Exception as e:
        print("❌ Error saat delete:", e)
        return False
