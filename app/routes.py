import logging
import os
from werkzeug.utils import secure_filename

from flask import Blueprint, request, jsonify, render_template
from app.rag import rag

"""
Main Routes Module.

Defines the core REST API endpoints for the application, including the front-end
rendering route, the chat query endpoint, and the chat history retrieval endpoint.
"""

logger = logging.getLogger(__name__)
main_bp = Blueprint('main', __name__)

# Configure upload folder (ensure this directory exists or create it dynamically)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@main_bp.route('/')
def home():
    """
    Serve the main front-end interface.

    Returns:
        Rendered HTML template for the chat interface.
    """
    return render_template('index.html')


@main_bp.route('/api/chat', methods=['POST'])
def chat():
    """
    Process a user chat query and return an AI-generated response.

    Expects a JSON payload containing 'prompt' (the user's question) and
    'session_id' (unique identifier for the chat session). It invokes the RAG
    pipeline to generate a context-aware answer.

    Returns:
        Response (JSON): A JSON object containing the 'response' text and 'session_id',
                         along with a 200 HTTP status code on success.
                         Returns a 400 status if missing parameters, or 500 on server errors.
    """

    user_input = request.form.get('prompt')
    session_id = request.form.get('session_id')
    file = request.files.get('file')
    if not user_input or not session_id:
        return jsonify({"error": "Prompt and session_id are required"}), 400

    file_path = None
    if file and file.filename != '':
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

    logger.info(f"Received query for session {session_id[:6]}...")

    try:
        response = rag.get_response(session_id, user_input, file_path=file_path)

        # Clean up the file from local disk immediately after processing
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        return jsonify({"response": response, "session_id": session_id}), 200
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")

        # Ensure cleanup happens even if an error occurs
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        return jsonify({"error": str(e)}), 500


@main_bp.route('/api/history/<session_id>', methods=['GET'])
def get_history(session_id):
    """
    Retrieve the chat history for a specific session.

    Fetches up to the last 50 messages from the MongoDB history manager and
    formats them for the front-end UI.

    Args:
        session_id (str): The unique identifier for the user's session.

    Returns:
        Response (JSON): A JSON object containing a 'history' list of message
                         dictionaries (with 'role' and 'content'), or a 500 status on error.
    """

    try:
        # Fetch up to 50 messages for the UI display
        history = rag.history_manager.get_history(session_id, limit=50)

        formatted_history = []
        for msg in history:
            # Check the type of Langchain message object to format it for JSON
            if msg.type == "human":
                formatted_history.append({"role": "user", "content": msg.content})
            elif msg.type == "ai":
                formatted_history.append({"role": "bot", "content": msg.content})

        return jsonify({"history": formatted_history}), 200
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        return jsonify({"error": str(e)}), 500

@main_bp.route('/api/history/<session_id>', methods=['DELETE'])
def delete_chat(session_id):

    """
    Endpoint to delete a specific chat session.

    Args:
        session_id (str): The unique identifier for the user's session.

    Returns:
        Response (JSON): A JSON object containing a success message with a 200 HTTP status code,
                             or an error message with a 500 HTTP status code upon failure.
    """
    try:
        rag.history_manager.delete_session(session_id)
        return jsonify({"message": f"Session {session_id} deleted."}), 200
    except Exception as e:
        logger.error(f"Error deleting history: {str(e)}")
        return jsonify({"error": str(e)}), 500