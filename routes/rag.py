from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from routes.chat_prompt import get_prompt_template, llm_model , groq_llm, groq_prompt_template ,ChatOllama_llm
from langchain.schema import Document
from dotenv import load_dotenv
from loggers import logging
import os

load_dotenv()

pinecone_api = os.getenv("PINECONE_API")
pc = Pinecone(api_key=pinecone_api)
index = pc.Index("isb-proj")
logging.info("Connected to Pinecone index: isb-proj")


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

logging.info("Initialized HuggingFace embeddings with model: sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

logging.info("Initialized Pinecone vector store with embeddings")

splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
logging.info("Initialized text splitter with chunk size 500 and overlap 50")

llm = llm_model()
logging.info("Initialized LLM model: Gemma 3 27b")

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
    
def Retrival_chain_rag(user_input):
    """
    Create a retrieval chain that uses the vector store to retrieve relevant documents.
    """
    retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5,"score_threshold": 0.7}
    )

    relevant_docs = retriever.get_relevant_documents(user_input)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    logging.info("Created retriever from Pinecone vector store")

    prompt = get_prompt_template(context=context,user_input=user_input)
    logging.info("getting the prompt template and retrive the context")

    llm = llm_model()
    logging.info("Initialized LLM model: Gemma 3 27b")

    return prompt , llm
    
def groq_retrival_chain():
    
    retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3,"score_threshold": 0.7}
    )
    logging.info("Created retriever from Pinecone vector store")

    llm = groq_llm()
    logging.info("LLM Added")

    prompt = groq_prompt_template()
    logging.info("Prompt template is added")

    combine_docs_chain = create_stuff_documents_chain(llm=llm,prompt=prompt)
    logging.info("Created document combination chain with groq llm and prompt template")

    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    logging.info("Created retrieval chain with retriever and document combination chain")

    return retrieval_chain



def ollama_retrival_chain():
    
    retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3,"score_threshold": 0.7}
    )
    logging.info("Created retriever from Pinecone vector store")

    llm = ChatOllama_llm()
    logging.info("LLM Added")

    prompt = groq_prompt_template()
    logging.info("Prompt template is added")

    combine_docs_chain = create_stuff_documents_chain(llm=llm,prompt=prompt)
    logging.info("Created document combination chain with groq llm and prompt template")

    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    logging.info("Created retrieval chain with retriever and document combination chain")

    return retrieval_chain