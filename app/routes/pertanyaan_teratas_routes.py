from fastapi import APIRouter

from app.controllers.data_controller import get_top_pertanyaan

top_pertanyaan_router = APIRouter()

@top_pertanyaan_router.get("")
def get_top_asked_questions():
    return get_top_pertanyaan()
