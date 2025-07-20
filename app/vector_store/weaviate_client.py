import os
import weaviate
from sentence_transformers import SentenceTransformer

# Lazy load the model
model = None

def get_model():
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

# Initialize Weaviate client using environment variables
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=f"https://{os.getenv('WEAVIATE_URL')}",
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv('WEAVIATE_API_KEY')),
    headers={"X-OpenAI-Api-Key": os.getenv('OPENAI_API_KEY', '')}
)

def store_bip_articles(articles, user_id):
    # Create schema if it doesn't exist
    if not client.collections.exists("BipArticle"):
        client.collections.create(
            name="BipArticle",
            properties=[
                {"name": "designation", "dataType": "text"},
                {"name": "unit", "dataType": "text"},
                {"name": "pu", "dataType": "number"},
                {"name": "lot", "dataType": "text"},
                {"name": "user_id", "dataType": "text"}
            ],
            vector_config={"vectorizer": "none"}
        )

    # Store articles with precomputed vectors
    with client.batch as batch:
        for article in articles:
            properties = {
                "designation": article["designation"],
                "unit": article["unit"],
                "pu": article.get("pu", 0.0),
                "lot": article.get("lot", ""),
                "user_id": user_id
            }
            batch.add_data_object(
                data_object=properties,
                collection="BipArticle",
                vector=article["vector"]
            )

def search_documents(user_id, query):
    model = get_model()
    query_vec = model.encode(query).tolist()
    response = client.query.get("BipArticle", ["designation", "unit", "pu", "lot"]).with_where({
        "path": ["user_id"],
        "operator": "Equal",
        "valueText": user_id
    }).with_near_vector({
        "vector": query_vec
    }).with_limit(1).do()
    
    hits = response["data"]["Get"]["BipArticle"]
    return hits[0] if hits else None
