import os
import weaviate
from sentence_transformers import SentenceTransformer
from weaviate.collections.classes.config import Property, DataType, Configure
from weaviate.auth import AuthApiKey

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
    auth_credentials=AuthApiKey(os.getenv('WEAVIATE_API_KEY')),
    headers={"X-OpenAI-Api-Key": os.getenv('OPENAI_API_KEY', '')}  # Optional
)

def store_bip_articles(articles, user_id):
    # Create schema if it doesn't exist
    if not client.collections.exists("BipArticle"):
       client.collections.create(
    name="BipArticle",
    properties=[
        Property(name="designation", data_type=DataType.TEXT),
        Property(name="unit", data_type=DataType.TEXT),
        Property(name="pu", data_type=DataType.NUMBER),
        Property(name="lot", data_type=DataType.TEXT),
        Property(name="user_id", data_type=DataType.TEXT),
    ],
    vector_config=Configure.Vectors.self_provided()
)

# Store articles with precomputed vectors
    bip_collection = client.collections.get("BipArticle")
    for article in articles:
        pu_value = article.get("pu", 0.0)
        try:
            pu_value = float(pu_value)
        except (ValueError, TypeError):
            pu_value = 0.0

        properties = {
            "designation": article.get("designation", ""),
            "unit": article.get("unit", ""),
            "pu": pu_value,
            "lot": article.get("lot", ""),
            "user_id": user_id
        }
        bip_collection.data.insert(
            properties,
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

    hits = response["data"]["Get"].get("BipArticle", [])
    return hits[0] if hits else None

def delete_bip_articles(user_id):
    # Supprime tous les objets BipArticle pour ce user_id
    client.collections.get("BipArticle").data.delete_many(
        where={
            "path": ["user_id"],
            "operator": "Equal",
            "valueText": user_id
        }
    )
