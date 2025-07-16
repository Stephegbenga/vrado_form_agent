# app.py
# Main Flask application file for a specialized Nigerian Business Registration AI Assistant.

import os
import uuid
import json
from datetime import datetime
from functools import wraps

import openai
from bson.objectid import ObjectId
from flask import Flask, jsonify, request, render_template_string, url_for, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.utils import secure_filename

# --- Configuration ---
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "chatbot_db"
# IMPORTANT: Add your OpenAI API Key here
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Database Connection (for logging conversations) ---
try:
    client = MongoClient(MONGO_URI)
    client.admin.command('ping')
    db = client[DB_NAME]
    messages_collection = db.messages
    print("MongoDB connection successful.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    db = None 

# --- OpenAI API Client ---
if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_API_KEY":
    openai.api_key = OPENAI_API_KEY
else:
    print("WARNING: OpenAI API Key is not set. The chatbot will not function.")
    openai = None

# --- Helper Functions ---

def get_openai_response(conversation_history):
    """
    Constructs a prompt and gets a response from the OpenAI API.
    """
    if not openai:
        return "I'm sorry, my connection to the AI service is not configured. Please contact support."

    # NEW PERSONA: This is the core prompt that defines the bot's new role.
    system_prompt = """
    You are a highly knowledgeable and professional AI assistant specializing in Nigerian business registration and corporate affairs. 
    Your name is 'CAC Connect'. Your expertise covers all aspects of setting up and managing a company in Nigeria, with a deep understanding of the Corporate Affairs Commission (CAC) processes.

    Your primary functions are:
    1.  **Answer Questions:** Provide clear, accurate, and up-to-date information about business registration types (Business Name, Limited Company, etc.), requirements, costs, timelines, and post-registration compliance.
    2.  **Guide Users:** Help users understand the steps involved in the CAC registration process. Explain complex legal and corporate terms in simple language.
    3.  **Provide Advice:** Offer general advice on choosing the right business structure and other related matters.
    
    Always be polite, professional, and encouraging. Start the first conversation by introducing yourself as 'CAC Connect' and asking how you can help with their business registration needs in Nigeria today.
    """
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "I'm sorry, I'm having trouble connecting to my brain right now. Please try again in a moment."

# --- Decorators ---
def db_connection_required(f):
    """Decorator to check for a valid DB connection."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if db is None:
            return jsonify({"error": "Database connection is not available."}), 503
        return f(*args, **kwargs)
    return decorated_function

# --- API Endpoints ---

@app.route("/")
def index():
    return "<h1>Nigerian Business Registration Chatbot Backend</h1>"

@app.route("/api/chat/<agent_id>", methods=['POST'])
@db_connection_required
def chat(agent_id):
    """Main endpoint for handling chat interactions."""
    data = request.json
    user_message = data.get("message")
    session_id = data.get("session_id")

    if not user_message or not session_id:
        return jsonify({"error": "Missing message or session_id"}), 400

    timestamp = datetime.utcnow()

    # 1. Save user's message to the log
    messages_collection.insert_one({
        "agent_id": agent_id,
        "session_id": session_id,
        "sender": "user",
        "message_text": user_message,
        "timestamp": timestamp
    })

    # 2. Get conversation history for context
    history_cursor = messages_collection.find({"session_id": session_id}).sort("timestamp", -1).limit(10)
    conversation_history = [
        {"role": "user" if msg["sender"] == "user" else "assistant", "content": msg["message_text"]}
        for msg in reversed(list(history_cursor))
    ]

    # 3. Get the bot's response from OpenAI
    bot_response_text = get_openai_response(conversation_history)

    # 4. Save bot's response to the log
    messages_collection.insert_one({
        "agent_id": agent_id,
        "session_id": session_id,
        "sender": "bot",
        "message_text": bot_response_text,
        "timestamp": datetime.utcnow()
    })
    
    return jsonify({"reply": bot_response_text})


if __name__ == '__main__':
    # Note: debug=True is not for production use.
    app.run(host='0.0.0.0', port=5000, debug=True)
