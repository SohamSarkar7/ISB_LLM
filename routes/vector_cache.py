import numpy as np
from pymongo import MongoClient
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from loggers import logging
import faiss
from bson import ObjectId


class VectorCache:
    def __init__(self, mongo_uri, db_name, collection_name="vector_cache"):
        self.client = MongoClient(mongo_uri)
        self.collection = self.client[db_name][collection_name]
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.dim = 384  # embedding dimension
        self.index = faiss.IndexFlatL2(self.dim)
        self.id_map = {}

        self._load_cache()

    def _load_cache(self):
        """Load all cached embeddings from MongoDB into FAISS."""
        docs = list(self.collection.find({}))
        if not docs:
            logging.info("No cache entries found in MongoDB.")
            return

        vectors = np.array([doc["embedding"] for doc in docs], dtype="float32")
        self.index.add(vectors)
        self.id_map = {i: str(doc["_id"]) for i, doc in enumerate(docs)}
        logging.info(f"Loaded {len(docs)} entries into FAISS cache.")

    def search_cache(self, query, threshold=0.75):
        """Search for a similar question in cache."""
        query_vec = self.embeddings.embed_query(query)
        query_np = np.array([query_vec], dtype="float32")

        if self.index.ntotal == 0:
            logging.info("Cache empty : calling LLM")
            return None

        D, I = self.index.search(query_np, k=1)
        best_idx = I[0][0]
        best_dist = D[0][0]

        # Convert L2 distance to similarity (approximation)
        similarity = 1 / (1 + best_dist)

        if similarity >= threshold:
            logging.info(f"Cache HIT : similarity={similarity:.4f}")
            cached_doc = self.collection.find_one(
                {"_id": ObjectId(self.id_map[best_idx])}
            )
            if cached_doc:
                return cached_doc["answer"]

        logging.info(f"Cache MISS : similarity={similarity:.4f}")
        return None

    def add_to_cache(self, session_id, question, answer):
        """Add a new Q&A to the cache."""
        vec = self.embeddings.embed_query(question)
        vec_np = np.array([vec], dtype="float32")

        inserted_id = self.collection.insert_one({
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "embedding": vec,
            "created_at": datetime.utcnow()
        }).inserted_id

        self.index.add(vec_np)
        self.id_map[self.index.ntotal - 1] = str(inserted_id)

        logging.info(f"Added to cache : Q: '{question[:50]}...'")
