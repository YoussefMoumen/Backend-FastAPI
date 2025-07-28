import os
import weaviate
from sentence_transformers import SentenceTransformer
from weaviate.collections.classes.config import Property, DataType, Configure
from weaviate.auth import AuthApiKey
from weaviate.classes.query import Filter

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

def store_dpgf_articles(articles, user_id):
    # Create schema if it doesn't exist
    if not client.collections.exists("DpgfArticle"):
        client.collections.create(
            name="DpgfArticle",
            properties=[
                Property(name="designation", data_type=DataType.TEXT),
                Property(name="unit", data_type=DataType.TEXT),
                Property(name="pu", data_type=DataType.NUMBER),
                Property(name="lot", data_type=DataType.TEXT),
                Property(name="user_id", data_type=DataType.TEXT),
            ],
            vector_config=Configure.Vectors.self_provided()
        )

    dpgf_collection = client.collections.get("DpgfArticle")
    for article in articles:
        # Get the pu value from the article and convert it to a float
        pu_value = article.get("pu", 0.0)
        try:
            pu_value = float(pu_value)
        except (ValueError, TypeError):
            pu_value = 0.0

        # Create a dictionary of properties to insert into the collection
        properties = {
            "designation": article.get("designation", ""),
            "unit": article.get("unit", ""),
            "pu": pu_value,
            "lot": article.get("lot", ""),
            "user_id": user_id
        }
        # Insert the properties and vector into the collection
        dpgf_collection.data.insert(
            properties,
            vector=article["vector"]
        )

# Define a function to search documents based on user_id and query
def search_documents(user_id, query):
    # Get the model
    model = get_model()
    # Encode the query
    query_vec = model.encode(query).tolist()
    # Get the BipArticle collection
    bip_collection = client.collections.get("BipArticle")
    # Query the collection with the encoded query and user_id
    results = bip_collection.query.near_vector(
        query_vec,
        where=Filter.by_property("user_id").equal(user_id),
        limit=1,
        return_properties=["designation", "unit", "pu", "lot"]
    )
    # Return the first hit if there are any, otherwise return None
    return results.objects[0].properties if results.objects else None


def delete_bip_articles(user_id):
    collection = client.collections.get("BipArticle")
    collection.data.delete_many(
        where=Filter.by_property("user_id").equal(user_id)
    )
    
def delete_dpgf_articles(user_id):
    collection = client.collections.get("DpgfArticle")
    collection.data.delete_many(
        where=Filter.by_property("user_id").equal(user_id)
    )