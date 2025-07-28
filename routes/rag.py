from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from routes.chat_prompt import get_prompt_template, llm_model
from langchain.schema import Document
from dotenv import load_dotenv
from loggers import logging
import os

load_dotenv()

pinecone_api = os.getenv("PINECONE_API")
pc = Pinecone(api_key=pinecone_api)
index = pc.Index("isb-proj")
logging.info("Connected to Pinecone index: isb-proj")


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

logging.info("Initialized HuggingFace embeddings with model: sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

logging.info("Initialized Pinecone vector store with embeddings")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
logging.info("Initialized text splitter with chunk size 500 and overlap 50")

llm = llm_model()
logging.info("Initialized LLM model: gemma3:1b")

prompt = get_prompt_template()
logging.info("Initialized prompt template for question-answering tasks")

def process_pdf_and_store_in_pinecone(file_path: str):
    try:
        # Load full PDF with metadata
        loader = PyPDFLoader(file_path)
        logging.info("Loaded PDF file: %s", file_path)
        docs = loader.load()
        logging.info("Loaded %d pages from PDF", len(docs))

        # Split documents
        split_docs = splitter.split_documents(docs)

        # Extract only 'page_content' and 'metadata'
        cleaned_docs = [
            Document(
                page_content=doc.page_content,
                

            )
            for doc in split_docs
        ]

        # Store only cleaned content+metadata
        added = vector_store.add_documents(cleaned_docs)
        logging.info("Added %d chunks to Pinecone index", len(added))

        return {
            "status": "success",
            "file": os.path.basename(file_path),
            "chunks_stored": len(added)
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
    
def Retrival_chain_rag():
    """
    Create a retrieval chain that uses the vector store to retrieve relevant documents.
    """
    retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2,"score_threshold": 0.7}
    )
    logging.info("Created retriever from Pinecone vector store")

    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    logging.info("Created document combination chain with LLM and prompt template")

    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    logging.info("Created retrieval chain with retriever and document combination chain")

    return retrieval_chain


    # Create a chain that retrieves relevant documents
    

