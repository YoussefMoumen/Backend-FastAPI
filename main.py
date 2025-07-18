from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.endpoints import upload_bip, upload_dpgf, analyze, export

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
