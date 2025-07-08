# app.py
# Main Flask application file for the AI data collection chatbot.

import os
import uuid
import json
from datetime import datetime
from functools import wraps
from dotenv import load_dotenv
load_dotenv()

import openai
from bson.objectid import ObjectId
from flask import Flask, jsonify, request, render_template_string, url_for, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.utils import secure_filename

# --- Configuration ---
# It's recommended to use environment variables for sensitive data in production.
# For example: os.environ.get('MONGO_URI')
# For this example, we'll define them here.
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "chatbot_db"
# IMPORTANT: Add your OpenAI API Key here
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for frontend interaction

# --- File Upload Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Ensure the upload folder exists

# --- Database Connection ---
try:
    client = MongoClient(MONGO_URI)
    # The following line triggers the connection attempt.
    client.admin.command('ping')
    db = client[DB_NAME]
    messages_collection = db.messages
    submissions_collection = db.form_submissions
    print("MongoDB connection successful.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    # Exit or handle the error appropriately if the DB is essential
    # For this script, we'll print the error and continue,
    # but endpoints will fail.
    db = None 

# --- OpenAI API Client ---
if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_API_KEY":
    openai.api_key = OPENAI_API_KEY
else:
    print("WARNING: OpenAI API Key is not set. The chatbot will not function.")
    openai = None

# --- Helper Functions ---

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_missing_fields(submission):
    """
    Analyzes a submission document to find which fields are still missing.
    This is crucial for guiding the chatbot's questions.
    """
    missing = []
    user_data = submission.get('user_data', {})
    
    # Define all required fields in the order they should be asked.
    required_text_fields = {
        "role": "your role (e.g., 'The main business owner')",
        "first_name": "your first name",
        "middle_name": "your middle name",
        "surname": "your surname",
        "date_of_birth": "your date of birth (YYYY-MM-DD)",
        "contact.phone_number": "your phone number",
        "contact.email_address": "your email address",
        "gender": "your gender",
        "address.residential": "your residential address (house number, street name, area)",
        "address.local_government": "your Local Government Area",
        "address.city": "your city",
        "address.state": "your state",
        "identification.id_type": "your ID type (e.g., NIN, Passport)",
        "identification.id_number": "your ID number",
        "share_details.number_of_shares": "the number of shares you hold"
    }
    
    # Check text fields
    for field, description in required_text_fields.items():
        # Handle nested fields
        keys = field.split('.')
        value = user_data
        try:
            for key in keys:
                value = value[key]
            if not value: # Check for empty strings etc.
                missing.append(description)
        except (KeyError, TypeError):
            missing.append(description)

    # Check file uploads
    file_uploads = submission.get('file_uploads', {})
    if not file_uploads.get('nin_image_url'):
        missing.append("a clear picture of your NIN")
    # FIX: Corrected key from 'passport_photo_url' to 'passport_image_url' to match the upload function
    if not file_uploads.get('passport_image_url'):
        missing.append("a clear copy of your passport photograph")
    if not file_uploads.get('signature_image_url'):
        missing.append("a picture of your signature on a plain sheet of paper")
        
    return missing

def get_openai_response(conversation_history, missing_fields):
    """
    Constructs a prompt and gets a response from the OpenAI API.
    """
    if not openai:
        return "I'm sorry, my connection to the AI service is not configured. Please contact support."

    # This is the core prompt that guides the AI's behavior
    system_prompt = f"""
    You are a friendly and professional AI assistant for a corporate registration service. 
    Your goal is to collect information from a user to complete their profile.
    Be polite, clear, and ask for ONE piece of information at a time.
    The current time is {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}. The user is in Ibadan, Oyo, Nigeria.
    
    Based on the conversation so far, you need to collect the following information:
    {', '.join(missing_fields)}.
    
    Please ask the user for the NEXT required item on the list.
    If the list is empty, congratulate the user on completing the form and tell them they can close the window.
    If the user asks a question, answer it politely and then return to asking for the next required piece of information.
    If the user says hello or starts the conversation, greet them warmly and ask for the first item on the list.
    Keep your responses concise.
    """
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)

    try:
        # UPGRADE: Using the more capable gpt-4o model.
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "I'm sorry, I'm having trouble connecting to my brain right now. Please try again in a moment."


def extract_and_update_data(session_id, conversation_history):
    """
    Uses OpenAI's function calling to extract structured data from the user's message
    and update the database. It uses the conversation history for context.
    """
    if not openai:
        return

    # This schema MUST match the MongoDB `form_submissions` schema
    function_schema = {
        "name": "update_form_submission",
        "description": "Updates the user's form submission with the extracted information.",
        "parameters": {
            "type": "object",
            "properties": {
                "role": {"type": "string", "description": "User's role, e.g., 'The main business owner'"},
                "first_name": {"type": "string"},
                "middle_name": {"type": "string"},
                "surname": {"type": "string"},
                "date_of_birth": {"type": "string", "description": "Date in YYYY-MM-DD format."},
                "phone_number": {"type": "string"},
                "email_address": {"type": "string"},
                "gender": {"type": "string", "enum": ["Male", "Female"]},
                "residential_address": {"type": "string", "description": "Full residential address including house number, street, and area."},
                "local_government": {"type": "string"},
                "city": {"type": "string"},
                "state": {"type": "string"},
                "id_type": {"type": "string"},
                "id_number": {"type": "string"},
                "number_of_shares": {"type": "integer"},
            },
            "required": [] # No fields are strictly required for a single function call
        }
    }

    extraction_system_prompt = """
    You are a data extraction assistant. Based on the provided conversation history, 
    analyze the LAST message from the user and extract any relevant information 
    that answers the bot's most recent question. Format the extracted data 
    according to the provided function.
    """
    messages_for_extraction = [{"role": "system", "content": extraction_system_prompt}]
    messages_for_extraction.extend(conversation_history)

    try:
        # UPGRADE: Using the more capable gpt-4o model.
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages_for_extraction,
            functions=[function_schema],
            function_call={"name": "update_form_submission"}
        )
        
        message = response.choices[0].message
        if message.function_call:
            arguments = json.loads(message.function_call.arguments)
            
            # Construct the update query for MongoDB using dot notation for nested fields
            update_data = {}
            if arguments.get("role"): update_data["user_data.role"] = arguments["role"]
            if arguments.get("first_name"): update_data["user_data.first_name"] = arguments["first_name"]
            if arguments.get("middle_name"): update_data["user_data.middle_name"] = arguments["middle_name"]
            if arguments.get("surname"): update_data["user_data.surname"] = arguments["surname"]
            if arguments.get("date_of_birth"): 
                try:
                    # Validate and convert date
                    dob = datetime.strptime(arguments["date_of_birth"], "%Y-%m-%d")
                    update_data["user_data.date_of_birth"] = dob
                except ValueError:
                    print(f"Invalid date format for {arguments['date_of_birth']}")
            if arguments.get("phone_number"): update_data["user_data.contact.phone_number"] = arguments["phone_number"]
            if arguments.get("email_address"): update_data["user_data.contact.email_address"] = arguments["email_address"]
            if arguments.get("gender"): update_data["user_data.gender"] = arguments["gender"]
            if arguments.get("residential_address"): update_data["user_data.address.residential"] = arguments["residential_address"]
            if arguments.get("local_government"): update_data["user_data.address.local_government"] = arguments["local_government"]
            if arguments.get("city"): update_data["user_data.address.city"] = arguments["city"]
            if arguments.get("state"): update_data["user_data.address.state"] = arguments["state"]
            if arguments.get("id_type"): update_data["user_data.identification.id_type"] = arguments["id_type"]
            if arguments.get("id_number"): update_data["user_data.identification.id_number"] = arguments["id_number"]
            if arguments.get("number_of_shares"): update_data["user_data.share_details.number_of_shares"] = int(arguments["number_of_shares"])
            
            if update_data:
                update_data["last_updated_at"] = datetime.utcnow()
                submissions_collection.update_one(
                    {"session_id": session_id},
                    {"$set": update_data}
                )
                print(f"Updated submission for session {session_id} with: {arguments}")

    except Exception as e:
        print(f"Error during data extraction: {e}")


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
    return "<h1>Chatbot Backend</h1><p>Navigate to /form/&lt;agent_id&gt; to see the frontend.</p>"

@app.route("/form/<agent_id>")
def form_page(agent_id):
    """Serves the main chat interface page."""
    try:
        with open("index.html", "r") as f:
            template_str = f.read()
        return render_template_string(template_str, agent_id=agent_id)
    except FileNotFoundError:
        return "index.html not found", 404

# This new endpoint is needed to serve the uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


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

    # 1. Save user's message
    messages_collection.insert_one({
        "agent_id": agent_id,
        "session_id": session_id,
        "sender": "user",
        "message_text": user_message,
        "timestamp": timestamp
    })

    # 2. Get conversation history right after saving the new message
    history_cursor = messages_collection.find({"session_id": session_id}).sort("timestamp", -1).limit(10)
    conversation_history = [
        {"role": "user" if msg["sender"] == "user" else "assistant", "content": msg["message_text"]}
        for msg in reversed(list(history_cursor))
    ]

    # 3. Find or create the form submission document
    submissions_collection.find_one_and_update(
        {"session_id": session_id},
        {"$setOnInsert": {
            "agent_id": agent_id,
            "session_id": session_id,
            "submission_status": "in_progress",
            "created_at": timestamp,
            "last_updated_at": timestamp,
            "user_data": {},
            "file_uploads": {}
        }},
        upsert=True
    )

    # 4. Attempt to extract structured data using the full context
    user_message_lower = user_message.lower()
    if "uploaded" not in user_message_lower and "hello" not in user_message_lower:
        extract_and_update_data(session_id, conversation_history)
    
    # 5. Re-fetch submission to get the latest updates
    submission = submissions_collection.find_one({"session_id": session_id})

    # 6. Determine what information is still missing
    missing_fields = get_missing_fields(submission)

    # 7. Get the bot's response from OpenAI (we already have the history)
    bot_response_text = get_openai_response(conversation_history, missing_fields)

    # 8. Save bot's response
    messages_collection.insert_one({
        "agent_id": agent_id,
        "session_id": session_id,
        "sender": "bot",
        "message_text": bot_response_text,
        "timestamp": datetime.utcnow()
    })
    
    # 9. If all fields are now complete, update status
    if not get_missing_fields(submissions_collection.find_one({"session_id": session_id})):
        submissions_collection.update_one(
            {"session_id": session_id},
            {"$set": {"submission_status": "completed", "last_updated_at": datetime.utcnow()}}
        )

    return jsonify({"reply": bot_response_text})


@app.route("/api/upload/<agent_id>", methods=['POST'])
@db_connection_required
def upload_file(agent_id):
    """Endpoint for handling file uploads."""
    session_id = request.form.get('session_id')
    file_type = request.form.get('file_type') # e.g., 'nin', 'passport', 'signature'
    
    if not all([session_id, file_type]):
        return jsonify({"error": "Missing session_id or file_type"}), 400
        
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        # Create a secure, unique filename
        filename = secure_filename(f"{session_id}_{file_type}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Generate a URL for the saved file
        file_url = url_for('uploaded_file', filename=filename, _external=True)

        # Update the corresponding field in the form_submissions collection
        # FIX: The key for the passport photo is 'passport_image_url' not 'passport_photo_url'
        update_key = f"file_uploads.{file_type}_image_url"
        result = submissions_collection.update_one(
            {"session_id": session_id},
            {"$set": {update_key: file_url, "last_updated_at": datetime.utcnow()}}
        )
        
        if result.matched_count == 0:
            # If session doesn't exist, create it.
            submissions_collection.insert_one({
                "agent_id": agent_id,
                "session_id": session_id,
                "submission_status": "in_progress",
                "created_at": datetime.utcnow(),
                "last_updated_at": datetime.utcnow(),
                "user_data": {},
                "file_uploads": {f"{file_type}_image_url": file_url}
            })
            
        return jsonify({"success": True, "message": "File uploaded successfully.", "file_url": file_url})
    else:
        return jsonify({"error": "File type not allowed"}), 400


if __name__ == '__main__':
    # Note: debug=True is not for production use.
    app.run(host='0.0.0.0', port=5000, debug=True)
