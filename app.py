from fastapi import FastAPI
from routes.llm_response import router as llm_router
from routes.llm_response import router as health_check
from routes.llm_response import router as upload_pdf


app = FastAPI()

# Include the LLM router
app.include_router(health_check, prefix="/api")
app.include_router(llm_router, prefix="/api")
app.include_router(upload_pdf, prefix="/api")
