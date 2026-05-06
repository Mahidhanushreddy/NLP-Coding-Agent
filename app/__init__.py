import logging
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

"""
Application Factory Module.

This module is responsible for bootstrapping the Flask application, configuring
production-level logging, enabling Cross-Origin Resource Sharing (CORS), and
registering application blueprints.
"""

def create_app():
    """
    Create and configure an instance of the Flask application.

    Loads environment variables, initializes the Flask app, sets up CORS for
    API access, configures basic logging for production monitoring, and registers
    the main routing blueprint.

    Returns:
        Flask: The configured Flask application instance.
    """

    load_dotenv(override=True)

    app = Flask(__name__)
    CORS(app)

    # Production Logging Setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )

    # Register Routes
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app