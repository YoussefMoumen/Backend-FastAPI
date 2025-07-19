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
    headers={"X-OpenAI-Api-Key": os.getenv('OPENAI_API_KEY', '')}  # Optional, add if needed
)

bip_vectors = {}

def index_documents(user_id, docs):
    model = get_model()
    bip_vectors[user_id] = [
        {"text": doc, "vector": model.encode(doc).tolist(), "pu": 100} for doc in docs
    ]

def search_documents(user_id, query):
    model = get_model()
    query_vec = model.encode(query).tolist()
    matches = bip_vectors.get(user_id, [])
    if not matches:
        return None

    best_match = max(
        matches,
        key=lambda doc: -sum([(a - b) ** 2 for a, b in zip(query_vec, doc["vector"])]),
    )
    return best_match