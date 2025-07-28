from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse
from langchain_ollama.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableSequence
from routes.chat_prompt import get_prompt_template, llm_model
from fastapi.responses import StreamingResponse
from pymongo import MongoClient
from routes.mongo_class import GroupedMongoChatHistory
from routes.schedular import start_scheduler
from fastapi.responses import JSONResponse
from loggers import logging
import json
import os
import shutil
from fastapi import FastAPI, UploadFile, File
from routes.rag import process_pdf_and_store_in_pinecone , Retrival_chain_rag
# Start daily memory flush
start_scheduler()

router = APIRouter()
mongo_uri = "mongodb://localhost:27017"
client = MongoClient(mongo_uri)
logging.info("Connected to MongoDB at %s", mongo_uri)

UPLOAD_FOLDER = "uploaded_files"
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



@router.post("/llm_response")
async def get_llm_response(user_input: str, session_id: str = Header(...)):
    """
    Stream LLM response using LangChain, MongoDB memory, and FastAPI.
    """
    try:
        # 1. Setup MongoDB-backed memory
        history = GroupedMongoChatHistory(
            session_id=session_id,
            mongo_uri=mongo_uri,
            db_name="llm_sessions",
            collection_name="chat_history"
        )

        memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True,
            chat_memory=history
        )

        memory.chat_memory.add_user_message(user_input)

        # 2. Prompt + LLM
        chain = Retrival_chain_rag()


        
        # async def stream_response():
        #     final_response = ""
        #     async for chunk in chain.astream({"input": user_input}):
        #         text = chunk.content
        #         final_response += text
        #         # Stream each chunk as a JSON line
        #         yield json.dumps({"response": text}) + "\n"

            
        #     memory.chat_memory.add_ai_message(final_response)

        # # 5. Stream JSON output to client
        # return StreamingResponse(stream_response(), media_type="application/jsonlines")

        response = chain.invoke({"input": user_input})

        final_response = response["answer"] 
        context_response = response['context'][0].page_content
        logging.info("LLM response generated successfully")  
        memory.chat_memory.add_ai_message(final_response)
        return JSONResponse(content={"response": final_response, "context": context_response})
    

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
