from fastapi import APIRouter
from app.controllers.data_controller import get_all_data, get_faq_detail_by_id, add_faq_to_chroma_and_json, update_faq_by_id,delete_faq_by_id
from pydantic import BaseModel
from fastapi import HTTPException
# from app.models.faq_model import FAQItem

data_router = APIRouter()

@data_router.get("")
def get_data():
    return get_all_data()

@data_router.get("/{id}")
def get_faq_by_id(id: str):
    try:
        detail = get_faq_detail_by_id(id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"FAQ dengan ID {id} tidak ditemukan.")

        return detail
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class FAQItem(BaseModel):
    pertanyaan: str
    jawaban: str
    kategori: str = ""
    keywords: str = ""

@data_router.post("")
def tambah_faq(item: FAQItem):
    success = add_faq_to_chroma_and_json(item.dict())
    if success:
        return {"message": "FAQ berhasil ditambahkan."}
    else:
        raise HTTPException(status_code=500, detail="Gagal menambahkan FAQ.")

@data_router.put("")
def update_faq(id: str, item: FAQItem):
    if update_faq_by_id(id, item.dict()):
        return {"message": f"FAQ dengan ID {id} berhasil diperbarui."}
    raise HTTPException(status_code=404, detail="FAQ tidak ditemukan.")

@data_router.delete("")
def delete_faq(id: str):
    if delete_faq_by_id(id):
        return {"message": f"FAQ dengan ID {id} berhasil dihapus."}
    raise HTTPException(status_code=404, detail="FAQ tidak ditemukan.")