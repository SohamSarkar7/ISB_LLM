from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableSequence
from pymongo import MongoClient
from routes.mongo_class import GroupedMongoChatHistory
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loggers import logging
import json
import os
import asyncio
import shutil
from fastapi import UploadFile, File
from routes.rag import process_pdf_and_store_in_pinecone , Retrival_chain_rag , groq_retrival_chain
from utils import check_session_limit , get_llm_choice
from routes.schedular import start_scheduler
from routes.vector_cache import VectorCache
from routes.agent import youtube_to_PDF
from dotenv import load_dotenv
load_dotenv()
# Start daily memory flush


start_scheduler()

# API ROUTER 
router = APIRouter(
    prefix="/api",      
    tags=["LLM Service"]  
)



#Mongo Setting
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
logging.info("Connected to MongoDB at %s", mongo_uri)


# Folder Structure
UPLOAD_FOLDER = os.path.join("uploaded_files", "PDF_documents")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Queue
MAX_CONCURRENT_USERS = 3          # max parallel processing
MAX_QUEUE_SIZE = 20               # max waiting users
request_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
semaphore = asyncio.Semaphore(MAX_CONCURRENT_USERS)


# Vector Cache
vector_cache = VectorCache(
    mongo_uri=mongo_uri,
    db_name="llm_sessions"
)


# Queue _Worker Function
async def queue_worker():
    """
    Worker that processes queued LLM requests.
    Each request is a tuple: (request_data, session_id, future)
    """
    while True:
        request_data, session_id, fut = await request_queue.get()
        try:
            async with semaphore:  # limit concurrent processing
                result = await process_llm_request(request_data, session_id)
                fut.set_result(result)
        except Exception as e:
            fut.set_exception(e)
        finally:
            request_queue.task_done()



for _ in range(MAX_CONCURRENT_USERS):
    asyncio.create_task(queue_worker())


## LLM_RESPONSE _FUNCTION
class LLMRequest(BaseModel):
    user_input: str


async def process_llm_request(request: LLMRequest, session_id: str):
    """
    Your existing /llm_response logic goes here, except:
    - Remove FastAPI's @router.post decorator
    - Remove return statements with StreamingResponse directly
    """
    user_input = request.user_input

    cached_answer = vector_cache.search_cache(user_input)
    if cached_answer:
        logging.info(f"Cache HIT — returning from cache for session={session_id}")
        return JSONResponse(
            content={
                "response": cached_answer,
                "cached": True,
                "source": "vector_cache"
            },
            status_code=200
        )

    if not check_session_limit(session_id):
        return JSONResponse(
            content={"response": "You've reached your 15 chats for today.\nCome back tomorrow to ask more questions."},
            status_code=200
        )

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
    llm_choice = get_llm_choice()
    final_response = ""

    if llm_choice == "primary":
        chain = groq_retrival_chain()

        def stream_response_groq(chain: RunnableSequence, user_input: str):
            nonlocal final_response
            for chunk in chain.stream({"input": user_input}):
                if "answer" in chunk:
                    text = chunk["answer"]
                    final_response += text
                    yield json.dumps({"response": text}) + "\n"
            memory.chat_memory.add_ai_message(final_response)
            vector_cache.add_to_cache(session_id, user_input, final_response)

        return StreamingResponse(
            stream_response_groq(chain=chain, user_input=user_input),
            media_type="application/jsonlines"
        )
    else:
        prompt, llm = Retrival_chain_rag(user_input=user_input)

        def stream_response(prompt, llm):
            nonlocal final_response
            for chunk in llm.stream(prompt):
                chunk_text = chunk.content
                final_response += chunk_text
                yield json.dumps({"response": chunk_text}) + "\n"
            memory.chat_memory.add_ai_message(final_response)
            vector_cache.add_to_cache(session_id, user_input, final_response)

        return StreamingResponse(
            stream_response(prompt=prompt, llm=llm),
            media_type="application/jsonlines"
        )




## ROUTER:- 
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
        logging.info("Youtube embedded successfully")
        return JSONResponse(content=result)
    except Exception as e:
        logging.error(f"Error processing YouTube link: {e}")
        raise HTTPException(status_code=400, detail=str(e))



# LLM_CLass and Route:-

@router.post("/llm_response")
async def llm_response(
    request: LLMRequest,
    session_id: str = Header(..., alias="session-id")
):
    # If queue is full, reject immediately
    if request_queue.full():
        return JSONResponse(
            status_code=429,
            content={
                "response": "Too many users are currently online. Please try again in a moment."
            }
        )

    # Create a future to hold the result
    fut = asyncio.get_event_loop().create_future()

    # Put into the queue
    await request_queue.put((request, session_id, fut))

    # Wait until it's processed
    return await fut

    