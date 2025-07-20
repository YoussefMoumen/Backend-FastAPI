from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import logging
from app.utils.extract_pdf import extract_text_from_pdf
from app.utils.extract_excel import extract_data_from_excel
from app.utils.extract_word import extract_text_from_word
from app.vector_store.weaviate_client import store_bip_articles, get_model
from sentence_transformers import SentenceTransformer
from app.utils.column_mapping import auto_map_fields  # À créer, voir ci-dessous

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@router.post("/upload_bip")
async def upload_bip(file: UploadFile = File(...), user_id: str = Form(...)):
    try:
        # Extract articles based on file type
        content = await file.read()
        if file.filename.endswith((".pdf", ".PDF")):
            articles = extract_text_from_pdf(content)  # Adjust to return structured data
        elif file.filename.endswith((".xlsx", ".xls", ".XLSX", ".XLS")):
            articles = extract_data_from_excel(content)
        elif file.filename.endswith((".docx", ".DOCX", ".doc", ".DOC")):
            articles = extract_text_from_word(content)  # Adjust to return structured data
        else:
            raise HTTPException(status_code=400, detail="Format non supporté. Utilisez PDF, Excel ou Word.")

        # --- AJOUT ICI : mapping automatique des colonnes ---
        articles = auto_map_fields(articles)  # <-- Cette fonction harmonise les clés

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
        store_bip_articles(vectorized_articles, user_id)
        logger.info(f"BIP uploaded and stored for user_id: {user_id} with {len(articles)} articles")

        # Return response
        return JSONResponse(content={
            "message": f"{len(articles)} articles indexés pour l'utilisateur {user_id}",
            "articles": articles,  # Original articles
            "vectorized_articles": vectorized_articles  # Articles with vectors
        })

    except HTTPException as http_err:
        logger.error(f"HTTP Error: {str(http_err)}")
        raise
    except Exception as e:
        logger.error(f"Error uploading BIP: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})