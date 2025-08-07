from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse
from langchain_ollama.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableSequence
from routes.chat_prompt import get_prompt_template, llm_model
from routes.agent import youtube_to_PDF
from fastapi.responses import StreamingResponse
from pymongo import MongoClient
from routes.mongo_class import GroupedMongoChatHistory
from routes.schedular import start_scheduler
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loggers import logging
import json
from datetime import datetime
import os
import shutil
from fastapi import FastAPI, UploadFile, File
from routes.rag import process_pdf_and_store_in_pinecone , Retrival_chain_rag , groq_retrival_chain
from utils import check_session_limit , get_llm_choice
from dotenv import load_dotenv
load_dotenv()
# Start daily memory flush


start_scheduler()

router = APIRouter(
    prefix="/api",      
    tags=["LLM Service"]  
)

mongo_uri = "mongodb+srv://sarkarsoham2002:1234@isbllm.ay4fqha.mongodb.net/?retryWrites=true&w=majority&appName=ISBLLM"
client = MongoClient(mongo_uri)
logging.info("Connected to MongoDB at %s", mongo_uri)

# Go up to ISB_LLM
UPLOAD_FOLDER = os.path.join("uploaded_files", "PDF_documents")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



@router.get("/")
def health_check():
    return {"status": "ok", "message": "Upload API is live."}



@router.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)

    # Save uploaded file locally
    try:
        with open(file_location, "wb") as buffer:

            shutil.copyfileobj(file.file, buffer)
            logging.info("File saved successfully at %s", file_location)
    except Exception as e:
        logging.error("File save failed: %s", str(e))
        return JSONResponse(status_code=500, content={"status": "error", "message": f"File save failed: {str(e)}"})
    

    # Call processing function
    result = process_pdf_and_store_in_pinecone(file_location)
    return JSONResponse(content=result)

class YouTubeLinkRequest(BaseModel):
    url: str

# Route
@router.post("/youtube_link_embedding")
async def youtube_video_link(request: YouTubeLinkRequest):
    try:
        pdf_path = youtube_to_PDF(request.url)
        result = process_pdf_and_store_in_pinecone(pdf_path)
        return JSONResponse(content=result)
    except Exception as e:
        logging.error(f"Error processing YouTube link: {e}")
        raise HTTPException(status_code=400, detail=str(e))

class LLMRequest(BaseModel):
    user_input: str

@router.post("/llm_response")
async def llm_response(
    request: LLMRequest,
    session_id: str = Header(..., alias="session-id")
):
    """
    Stream LLM response using LangChain, MongoDB memory, and FastAPI.
    """

    # if not can_process_immediately():
    #     return JSONResponse(
    #         content={"response": "⚠️ Too many users right now. Please wait..."},
    #         status_code=429
    #     )
    
    try:
        user_input = request.user_input

        if not check_session_limit(session_id):
            return JSONResponse(
                content={"response": "❗ You’ve reached your 10 chats for today.\nCome back tomorrow to ask more questions."},
                status_code=200
            )

        # 1. Setup MongoDB-backed memory
        history = GroupedMongoChatHistory(
            session_id=session_id,
            mongo_uri=mongo_uri,  # Make sure this is defined in your module
            db_name="llm_sessions",
            collection_name="chat_history"
        )

        memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True,
            chat_memory=history
        )

        memory.chat_memory.add_user_message(user_input)

        # 2. Setup retrieval chain
        llm_choice = get_llm_choice()


        if llm_choice == "primary":

            chain = groq_retrival_chain()

            def stream_response_groq(chain: RunnableSequence,user_input: str):
                final_response = ""
                for chunk in chain.stream({"input": user_input}):
                    if "answer" in chunk:
                        text = chunk["answer"]
                        final_response += text
                        # Stream each chunk as a JSON line
                        yield json.dumps({"response": text}) + "\n"
                memory.chat_memory.add_ai_message(final_response)

            return StreamingResponse(stream_response_groq(chain=chain,user_input=user_input), media_type="application/jsonlines")
            
        else:
            prompt, llm = Retrival_chain_rag(user_input=user_input)

            def stream_response(prompt,llm):
                final_response = ""
                for chunk in llm.stream(prompt):
                    chunk_text = chunk.content
                    final_response += chunk_text
                    # Stream each chunk as a JSON line
                    yield json.dumps({"response": chunk_text}) + "\n"
                memory.chat_memory.add_ai_message(final_response)
            return StreamingResponse(stream_response(prompt=prompt,llm = llm), media_type="application/jsonlines")
    
    except Exception as e:
        logging.error(f"Error in get_llm_response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

    # finally:
    #     remove_user()