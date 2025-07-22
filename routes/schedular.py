from apscheduler.schedulers.background import BackgroundScheduler
from pymongo import MongoClient
import datetime

mongo_uri = "mongodb://localhost:27017"
client = MongoClient(mongo_uri)
collection = client["llm_sessions"]["chat_history"]

def clear_all_sessions():
    print(f"Clearing all sessions at {datetime.datetime.now()}")
    collection.delete_many({})

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(clear_all_sessions, trigger='cron', hour=0, minute=0)  # Every day at 12 AM
    scheduler.start()
