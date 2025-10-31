from fastapi import FastAPI
from app.routes.ask_routes import ask_router
from app.routes.data_routes import data_router
from app.routes.count_routes import counter_router
from app.routes.pertanyaan_teratas_routes import top_pertanyaan_router
from app.routes.topic_routes import topic_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Roxy AI - TEST Gold Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ INI BAHAYA untuk PRODUKSI
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ask_router, prefix="/ask", tags=["Tanya Jawab"])
app.include_router(data_router, prefix="/data", tags=["Data"])
app.include_router(counter_router, prefix="/count", tags=["Data"])
app.include_router(top_pertanyaan_router, prefix="/teratas", tags=["Data"])
app.include_router(topic_router, prefix="/topic", tags=["Data"])

@app.get("/")
def root():
    return {"message": "Roxy AI API is running"}
