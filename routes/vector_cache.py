import numpy as np
from pymongo import MongoClient
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings 
from loggers import logging
import faiss

class VectorCache:
    def __init__(self, mongo_uri, db_name, collection_name="vector_cache"):
        self.client = MongoClient(mongo_uri)
        self.collection = self.client[db_name][collection_name]
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # adjust model
        self.index = None
        self.id_map = {}

        self._load_cache()

    def _load_cache(self):
        """Load all cached embeddings from MongoDB into FAISS."""
        docs = list(self.collection.find({}))
        if not docs:
            self.index = faiss.IndexFlatL2(384)  
            return
        
        vectors = np.array([doc["embedding"] for doc in docs], dtype="float32")
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)
        self.id_map = {i: str(doc["_id"]) for i, doc in enumerate(docs)}

    def search_cache(self, query, threshold=0.75):
        """Search for a similar question in cache."""
        query_vec = self.embeddings.embed_query(query)
        query_np = np.array([query_vec], dtype="float32")

        if self.index.ntotal == 0:
            logging.info("Cache empty — calling LLM")
            return None
        
        D, I = self.index.search(query_np, k=1)
        if len(I) > 0 and D[0][0] < (1 - threshold):  
            logging.info(f"Cache HIT — score={1-D[0][0]:.4f}")
            cached_doc = self.collection.find_one({"_id": self.id_map[I[0][0]]})
            
            if cached_doc:
                return cached_doc["answer"]
        logging.info(f"Cache MISS — best score={1-D[0][0]:.4f}")
        return None

    def add_to_cache(self, session_id, question, answer):
        """Add a new Q&A to the cache."""
        vec = self.embeddings.embed_query(question)
        vec_np = np.array([vec], dtype="float32")
        self.index.add(vec_np)

        result = self.collection.insert_one({
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "embedding": vec,
            "created_at": datetime.utcnow()
        })
        self.id_map[len(self.id_map)] = result.inserted_id
