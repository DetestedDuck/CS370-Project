import torch
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from transformers import AutoModel, AutoTokenizer

MONGO_URI = "mongodb://mongodb:27017/"
DB_NAME = "rag_data"
COLLECTION_RAW = "raw_data"
COLLECTION_PROCESSED = "processed_data"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "rag_embeddings"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
raw_collection = db[COLLECTION_RAW]
processed_collection = db[COLLECTION_PROCESSED]

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Helper Function: Generate Embeddings
def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

# Featurization Pipeline
def featurize_and_store():
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    raw_data = list(raw_collection.find({"processed": {"$ne": True}}))
    if not raw_data:
        print("No new raw data to process.")
        return

    for record in raw_data:
        text = record.get("text") or record.get("url", "Unknown")
        embedding = generate_embeddings(text)

        point = PointStruct(id=record["_id"], vector=embedding, payload={"source": record["source"]})
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point])

        processed_record = {
            "_id": record["_id"],
            "text": text,
            "embedding_id": str(record["_id"]),
            "source": record["source"],
            "processed": True
        }
        processed_collection.insert_one(processed_record)

    print(f"Processed and stored {len(raw_data)} records.")

if __name__ == "__main__":
    featurize_and_store()
