from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
vector_db = Chroma(persist_directory="db_faq_baru_3", embedding_function=embedding_model)

# Ambil semua dokumen dari DB
all_docs = vector_db.get()

# Tampilkan isi
for i in range(len(all_docs['documents'])):
    print(f"\n🔢 Data #{i+1}")
    print(f"📝 Jawaban : {all_docs['documents'][i]}")
    print(f"❓ Pertanyaan : {all_docs['metadatas'][i].get('question')}")
    print(f"📚 kategori : {all_docs['metadatas'][i].get('topic')}")
    print(f"📚 keyword : {all_docs['metadatas'][i].get('topic')}")
