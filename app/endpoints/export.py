from fastapi import APIRouter
from fastapi.responses import FileResponse
import pandas as pd
from tempfile import NamedTemporaryFile
from app.endpoints.upload_dpgf import dpgf_store

router = APIRouter()

@router.get("/export/{user_id}")
def export(user_id: str):
    df = dpgf_store.get(user_id)
    if df is None:
        return {"error": "DPGF non trouvé"}

    tmp = NamedTemporaryFile(delete=False, suffix=".xlsx")
    df.to_excel(tmp.name, index=False)
    return FileResponse(tmp.name, filename="DPGF_avec_prix.xlsx")
