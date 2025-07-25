from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import logging
from app.utils.extract_pdf import extract_text_from_pdf
from app.utils.extract_excel import dpgf_extract_data_from_excel
from app.utils.extract_word import extract_text_from_word
from app.vector_store.weaviate_client import store_dpgf_articles, get_model, delete_dpgf_articles
from sentence_transformers import SentenceTransformer
# from app.utils.column_mapping import auto_map_fields
import json

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@router.post("/upload_dpgf")
async def upload_dpgf(file: UploadFile = File(...), user_id: str = Form(...), columns_map: str = Form(None)):
    content = await file.read()
    articles = []
    if file.filename.endswith((".xlsx", ".xls", ".XLSX", ".XLS")):
        columns_dict = None
        if columns_map:
            logger.info(f"columns_map reçu: {columns_map}")
            try:
                columns_dict = json.loads(columns_map)
            except Exception as e:
                logger.error(f"Erreur parsing columns_map: {e}")
                raise HTTPException(status_code=400, detail="columns_map JSON invalide")
        articles = dpgf_extract_data_from_excel(content, columns_dict)
    elif file.filename.endswith((".pdf", ".PDF")):
        articles = extract_text_from_pdf(content)  # Adjust to return structured data
    elif file.filename.endswith((".docx", ".DOCX", ".doc", ".DOC")):
        articles = extract_text_from_word(content)  # Adjust to return structured data
    else:
        raise HTTPException(status_code=400, detail="Format non supporté. Utilisez PDF, Excel ou Word.")

    # --- AJOUT ICI : mapping automatique des colonnes ---
    # articles = auto_map_fields(articles)  # <-- Cette fonction harmonise les clés

    # Ensure articles is a list of dictionaries with required fields
    if not isinstance(articles, list):
        articles = [articles] if isinstance(articles, dict) else []
    for article in articles:
        if not all(key in article for key in ["designation", "unit", "pu", "lot"]):
            article.update({"unit": "", "pu": 0.0, "lot": ""})  # Default values if missing

    if not articles or not isinstance(articles, list):
        logger.error("No valid articles extracted from file.")
        raise HTTPException(status_code=400, detail="Aucun article valide trouvé dans le fichier.")

    # Vectorize articles
    model = get_model()
    vectorized_articles = []
    for article in articles:
        if not isinstance(article, dict) or "designation" not in article:
            logger.warning(f"Article missing 'designation': {article}")
            continue
        designation_vector = model.encode(article["designation"]).tolist()
        vectorized_article = article.copy()
        vectorized_article["vector"] = designation_vector
        vectorized_articles.append(vectorized_article)

    # Store in Weaviate
    store_dpgf_articles(vectorized_articles, user_id)
    logger.info(f"DPGF uploaded and stored for user_id: {user_id} with {len(articles)} articles")

    # Return response
    return JSONResponse(content={
        "message": f"{len(articles)} articles indexés pour l'utilisateur {user_id}",
        "articles": articles,  # Original articles
        "vectorized_articles": vectorized_articles  # Articles with vectors
    })

@router.delete("/delete_dpgf")
async def delete_dpgf(user_id: str):
    delete_dpgf_articles(user_id)
    return {"message": f"Articles DPGF supprimés pour l'utilisateur {user_id}"}