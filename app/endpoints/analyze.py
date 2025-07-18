from fastapi import APIRouter, File, UploadFile, Form
import pandas as pd
from tempfile import NamedTemporaryFile

router = APIRouter()
dpgf_store = {}

@router.post("/upload_dpgf")
async def upload_dpgf(file: UploadFile = File(...), user_id: str = Form(...)):
    tmp = NamedTemporaryFile(delete=False)
    tmp.write(await file.read())
    tmp.close()
    df = pd.read_excel(tmp.name)
    dpgf_store[user_id] = df
    return {"message": f"DPGF stocké pour {user_id}"}
