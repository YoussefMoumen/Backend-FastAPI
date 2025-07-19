from fastapi import APIRouter, File, UploadFile, Form
from app.utils.extract_pdf import extract_text_from_pdf
from app.utils.extract_excel import extract_data_from_excel
from app.utils.extract_word import extract_text_from_word
from app.vector_store.weaviate_client import index_documents

router = APIRouter()

@router.post("/upload_bip")
async def upload_bip(file: UploadFile = File(...), user_id: str = Form(...)):
    if file.filename.endswith((".pdf", ".PDF")):
        docs = extract_text_from_pdf(file.file)
    elif file.filename.endswith((".xlsx", ".xls", ".XLSX", ".XLS")):
        docs = extract_data_from_excel(file.file)
    elif file.filename.endswith((".docx", ".DOCX", ".doc", ".DOC")):
        docs = extract_text_from_word(file.file)
    else:
        return {"error": "Format non supporté"}

    # Ensure docs is a list of strings
    if isinstance(docs, dict):
        docs = list(docs.values())
    elif not isinstance(docs, list):
        docs = [str(docs)]
    else:
        docs = [str(d) for d in docs]

    index_documents(user_id, docs)
    return {"message": f"{len(docs)} articles indexés pour l'utilisateur {user_id}"}
