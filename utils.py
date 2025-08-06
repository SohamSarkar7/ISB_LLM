import json
from langchain_core.runnables import RunnableSequence
from datetime import datetime
from pymongo import MongoClient


mongo_uri = "mongodb+srv://sarkarsoham2002:1234@isbllm.ay4fqha.mongodb.net/?retryWrites=true&w=majority&appName=ISBLLM"
client = MongoClient(mongo_uri)
MAX_DAILY_CALLS = 14400
usage_collection = client["llm_sessions"]["usage_counter"]

def stream_response(chain: RunnableSequence,user_input: str):
    final_response = ""
    for chunk in chain.stream({"input": user_input}):
        if "answer" in chunk:
            text = chunk["answer"]
            final_response += text
            # Stream each chunk as a JSON line
            yield json.dumps({"response": text}) + "\n"


def stream_response_groq(chain: RunnableSequence,user_input: str):
    final_response = ""
    for chunk in chain.stream({"input": user_input}):
        if "answer" in chunk:
            text = chunk["answer"]
            final_response += text
            # Stream each chunk as a JSON line
            yield json.dumps({"response": text}) + "\n"


def get_llm_choice():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    usage_doc = usage_collection.find_one({"date": today})
    
    if usage_doc:
        if usage_doc["llm_calls"] >= MAX_DAILY_CALLS:
            return "fallback"
        else:
            usage_collection.update_one({"date": today}, {"$inc": {"llm_calls": 1}})
            return "primary"
    else:
        usage_collection.insert_one({"date": today, "llm_calls": 1})
        return "primary"
                
