from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.endpoints import upload_bip, upload_dpgf, analyze, export
import os
import uvicorn
import logging

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_bip.router)
app.include_router(upload_dpgf.router)
app.include_router(analyze.router)
app.include_router(export.router)

@app.get("/")
def root():
    return {"message": "API IA BIP-DPGF est active"}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    port = int(os.getenv("PORT", 10000))
    logger.info(f"Starting server on host 0.0.0.0 and port {port}")
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)