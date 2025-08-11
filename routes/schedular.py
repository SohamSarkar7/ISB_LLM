from apscheduler.schedulers.background import BackgroundScheduler
from pymongo import MongoClient
import datetime

mongo_uri = "mongodb+srv://sarkarsoham2002:1234@isbllm.ay4fqha.mongodb.net/?retryWrites=true&w=majority&appName=ISBLLM"
client = MongoClient(mongo_uri)
collection = client["llm_sessions"]["chat_history"]

def clear_all_sessions():
    print(f"Clearing all sessions at {datetime.datetime.now()}")
    collection.drop()

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(clear_all_sessions, trigger='cron', hour=0, minute=0)

    scheduler.start()
