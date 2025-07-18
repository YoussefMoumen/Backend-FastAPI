from fastapi import APIRouter, File, UploadFile, Form
from app.utils.extract_pdf import extract_text_from_pdf
from app.utils.extract_excel import extract_data_from_excel
from app.utils.extract_word import extract_text_from_word
from app.vector_store.weaviate_client import index_documents

router = APIRouter()

@router.post("/upload_bip")
async def upload_bip(file: UploadFile = File(...), user_id: str = Form(...)):
    if file.filename.endswith(".pdf"):
        docs = extract_text_from_pdf(file.file)
    elif file.filename.endswith(".xlsx"):
        docs = extract_data_from_excel(file.file)
    elif file.filename.endswith(".docx"):
        docs = extract_text_from_word(file.file)
    else:
        return {"error": "Format non supporté"}

    index_documents(user_id, docs)
    return {"message": f"{len(docs)} articles indexés pour l'utilisateur {user_id}"}
