from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(model="roxy-ai", temperature=0.0)

rephrase_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "Tolong tulis ulang jawaban ini agar lebih jelas dan menarik terkait produk resmi TEST gold, serta mudah dipahami oleh pelanggan awam."
     "Jawab menggunakan bahasa yang lugas tanpa menjelaskan apapun selain dari pengembangan data tersebut pakailah bahasa indonesia."),
    ("human", 
     "Berikut adalah jawaban mentah yang ditemukan:\n\n{answer_raw}\n\n")
])
rephrase_chain = rephrase_prompt | llm

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "Kamu adalah Roxy AI, asisten virtual resmi dari TEST Gold. "
     "Jawaban kamu harus profesional, jelas, akurat, dan selalu dalam bahasa Indonesia. "
     "Jawaban HANYA boleh berdasarkan data dari RAG (`db_faq_baru_3`)."),
    ("human", 
     "Berikut adalah data hasil pencarian:\n\n{context}\n\n"
     "Pertanyaan: {question}\n\n")
])
chat_chain = chat_prompt | llm
