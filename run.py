from app import create_app

app = create_app()

if __name__ == '__main__':
    """
    Application Entry Point.

    Serves as the execution script to start the Flask server on port 5000."""
    app.run(host='0.0.0.0', port=5000 )