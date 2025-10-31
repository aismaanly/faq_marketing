from fastapi import APIRouter
from app.controllers.data_controller import all_topic

topic_router = APIRouter()

@topic_router.get("")
def get_all_topics():
    return all_topic()
