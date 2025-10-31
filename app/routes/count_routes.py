from fastapi import APIRouter
from app.utils.ask_counter import get_all_counts

counter_router = APIRouter()

@counter_router.get("")
def get_question_stats():
    return get_all_counts()
