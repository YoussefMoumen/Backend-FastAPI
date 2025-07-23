from fastapi import APIRouter, UploadFile, File
from app.utils.extract_excel import extract_columns_and_reverse_map

router = APIRouter()

@router.post("/bip_column_extract")
async def bip_column_extract(file: UploadFile = File(...)):
    file_bytes = await file.read()
    result = extract_columns_and_reverse_map(file_bytes)
    return result