from fastapi import FastAPI
from routes.llm_response import router as llm_router



app = FastAPI()

# Include the LLM router
app.include_router(llm_router)
