import os
import logging
from pymongo import MongoClient
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

"""
MongoDB Chat History Manager.

Handles the persistent storage and retrieval of conversational history using
MongoDB. Ensures context windows do not exceed specified limits.
"""

class MongoHistoryManager:
    """
    Manager for interacting with the MongoDB chat history collection.
    """

    def __init__(self):
        # Initialize connection using the URI from .env
        mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
        self.client = MongoClient(mongo_uri)
        self.db = self.client["rag_agent_db"]
        self.collection = self.db["chat_histories"]
        logger.info("Connected to MongoDB for chat history.")

    def get_history(self, session_id, limit=10):
        """
        Retrieve the most recent messages for a session.

        Args:
            session_id (str): The unique identifier for the chat session.
            limit (int, optional): The maximum number of recent messages to return. Defaults to 10.

        Returns:
            list: A list of LangChain Message objects (HumanMessage, AIMessage).
        """

        record = self.collection.find_one({"session_id": session_id})

        if not record:
            return []

        # Convert dict objects back to LangChain HumanMessage/AIMessage
        chat_history = []
        messages = record.get("messages", [])[-limit:]

        for msg in messages:
            if msg["type"] == "human":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["type"] == "ai":
                chat_history.append(AIMessage(content=msg["content"]))

        return chat_history

    def save_interaction(self, session_id, human_input, ai_response, limit=10):
        """
        Save a user-AI interaction to the database and trim the history.

        Appends the new exchange to the session's record and enforces a rolling
        window by keeping only the most recent `limit` messages to prevent context overflow.

        Args:
            session_id (str): The unique identifier for the chat session.
            human_input (str | list): The user's input.
            ai_response (str | list): The AI's response.
            limit (int, optional): The maximum number of messages to retain. Defaults to 10.
        """
        def ensure_string(data):
            if isinstance(data, list):
                return "".join([str(item.get('text', item)) if isinstance(item, dict) else str(item) for item in data])
            return str(data)

        new_messages = [
            {"type": "human", "content": ensure_string(human_input)},
            {"type": "ai", "content": ensure_string(ai_response)}
        ]
        # 1. Upsert (insert or update) the new messages
        self.collection.update_one(
            {"session_id": session_id},
            {"$push": {"messages": {"$each": new_messages}}},
            upsert=True
        )

        # 2. Enforce memory limit to keep the context window from overflowing
        record = self.collection.find_one({"session_id": session_id}, {"messages": 1})
        if record and len(record.get("messages", [])) > limit:
            recent_messages = record["messages"][-limit:]
            self.collection.update_one(
                {"session_id": session_id},
                {"$set": {"messages": recent_messages}}
            )

    def delete_session(self, session_id):
        """
        Hard delete a chat session from the database.

        Args:
            session_id (str): The unique identifier for the chat session to be deleted.
        """
        try:
            self.collection.delete_one({"session_id": session_id})
            logger.info(f"Session {session_id} deleted from MongoDB.")
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")