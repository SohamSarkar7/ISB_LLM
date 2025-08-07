import json
from langchain_core.runnables import RunnableSequence
from datetime import datetime
from pymongo import MongoClient
import redis


MAX_CONCURRENT_USERS = 500
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

mongo_uri = "mongodb+srv://sarkarsoham2002:1234@isbllm.ay4fqha.mongodb.net/?retryWrites=true&w=majority&appName=ISBLLM"
client = MongoClient(mongo_uri)
MAX_DAILY_CALLS = 14400
usage_collection = client["llm_sessions"]["usage_counter"]

session_counter_collection = client["llm_sessions"]["session_counters"]
SESSION_CALL_LIMIT = 15

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
    

def check_session_limit(session_id: str) -> bool:
    today = datetime.utcnow().strftime("%Y-%m-%d")

    doc = session_counter_collection.find_one({"session_id": session_id, "date": today})

    if doc:
        if doc["llm_calls"] >= SESSION_CALL_LIMIT:
            return False  # Limit reached
        else:
            session_counter_collection.update_one(
                {"session_id": session_id, "date": today},
                {"$inc": {"llm_calls": 1}}
            )
            return True
    else:
        session_counter_collection.insert_one({
            "session_id": session_id,
            "date": today,
            "llm_calls": 1
        })
        return True

def can_process_immediately() -> bool:
    active_users = redis_client.get("active_users")
    if not active_users:
        redis_client.set("active_users", 1)
        return True
    elif int(active_users) < MAX_CONCURRENT_USERS:
        redis_client.incr("active_users")
        return True
    else:
        return False

def remove_user():
    redis_client.decr("active_users")
                
