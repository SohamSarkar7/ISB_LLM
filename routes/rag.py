# rag_service.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
import logging

load_dotenv()

# Global one-time setup
pinecone_api = os.getenv("PINECONE_API")
pc = Pinecone(api_key=pinecone_api)
index = pc.Index("isb-proj")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Main processing function
def process_pdf_and_store_in_pinecone(file_path: str):
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if not docs:
            return {"status": "failed", "message": "No content found."}

        chunks = splitter.split_documents(docs)
        added_docs = vector_store.add_documents(chunks)

        return {
            "status": "success",
            "filename": os.path.basename(file_path),
            "chunks_stored": len(added_docs)
        }

    except Exception as e:
        logging.exception("PDF processing failed.")
        return {"status": "error", "message": str(e)}
