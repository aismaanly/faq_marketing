from fastapi import APIRouter
from app.models.question_model import QuestionRequest
from app.controllers.ask_controller import process_question

ask_router = APIRouter()

@ask_router.post("")
def ask_roxy(request: QuestionRequest):
    return process_question(request.question)
