from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from pymongo import MongoClient
from typing import List

class GroupedMongoChatHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, mongo_uri: str, db_name: str, collection_name: str):
        self.session_id = session_id
        self.client = MongoClient(mongo_uri)
        self.collection = self.client[db_name][collection_name]
        self.collection.update_one(
            {"session_id": session_id},
            {"$setOnInsert": {"messages": [], "session_id": session_id}},
            upsert=True
        )

    @property
    def messages(self) -> List[BaseMessage]:
        doc = self.collection.find_one({"session_id": self.session_id})
        if not doc or "messages" not in doc:
            return []
        return messages_from_dict(doc["messages"])

    def add_message(self, message: BaseMessage) -> None:
        # Proper serialization using LangChain utility
        serialized = messages_to_dict([message])
        self.collection.update_one(
            {"session_id": self.session_id},
            {"$push": {"messages": {"$each": serialized}}},
        )

    def get_messages(self) -> List[BaseMessage]:
        return self.messages

    def clear(self) -> None:
        self.collection.delete_one({"session_id": self.session_id})
