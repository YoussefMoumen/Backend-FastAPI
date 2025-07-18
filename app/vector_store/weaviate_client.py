import weaviate
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = weaviate.Client("vnnojers1slcuab06h2rg.c0.europe-west3.gcp.weaviate.cloud")

bip_vectors = {}

def index_documents(user_id, docs):
    bip_vectors[user_id] = [
        {"text": doc, "vector": model.encode(doc).tolist(), "pu": 100} for doc in docs
    ]

def search_documents(user_id, query):
    query_vec = model.encode(query).tolist()
    matches = bip_vectors.get(user_id, [])
    if not matches:
        return None

    best_match = max(
        matches,
        key=lambda doc: -sum([(a - b) ** 2 for a, b in zip(query_vec, doc["vector"])]),
    )
    return best_match
