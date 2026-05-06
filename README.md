# NLP Coding Agent (RAG-Powered)

An advanced, full-stack Flask application that serves as an AI developer assistant. This application utilizes Retrieval-Augmented Generation (RAG) to provide context-aware programming assistance. It features a modern web interface, automated quality assurance for AI responses, persistent chat history, and document ingestion capabilities.

## 🚀 Features

*   **Conversational AI:** Powered by Google's `gemini-3-flash-preview` for high-quality, fast generation.
*   **Retrieval-Augmented Generation (RAG):** Uses `Pinecone` as a vector database and `GoogleGenerativeAIEmbeddings` to retrieve relevant document context for answering queries.
*   **Automated Quality Assurance:** Includes a deterministic LLM evaluator (temperature = 0.0) that strictly grades draft responses for relevance, safety, professionalism, and formatting before displaying them to the user.
*   **Persistent Chat Memory:** Chat sessions are saved and retrieved using **MongoDB**, enforcing a rolling context window to prevent token overflow.
*   **Dynamic File Uploads:** Users can temporarily attach `.pdf` or `.txt` files directly in the UI to provide immediate context to their queries[.
*   **Batch Document Ingestion:** A standalone pipeline to parse, chunk, embed, and upload local Word documents (`.docx`) to the Pinecone index.
*   **Modern Frontend:** A responsive, dark-themed UI built with HTML/CSS/JS, featuring syntax highlighting via `highlight.js` and Markdown rendering.

## 🛠️ Tech Stack

*   **Backend:** Python, Flask, Gunicorn
*   **AI/Orchestration:** LangChain, Google Generative AI (Gemini)
*   **Vector Database:** Pinecone (`langchain-pinecone`)
*   **Standard Database:** MongoDB (`pymongo`)
*   **Frontend:** HTML5, CSS3, Vanilla JavaScript, Marked.js, Highlight.js

## ⚙️ Prerequisites

Before running the application, ensure you have the following installed and configured:
1.  **Python 3.8+**
2.  **MongoDB:** Running locally on port `27017` or a cloud MongoDB Atlas URI.
3.  **API Keys:**
    *   Google Gemini API Key
    *   Pinecone API Key & Environment/Index Name

## 📦 Installation

**1. Clone the repository and navigate to the project directory:**
```bash
git clone <your-repo-url>
cd <your-project-folder>
```

**2. Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Environment Variables:**
Create a `.env` file in the root directory and add the following keys. **Do not commit this file to version control.**

```env
GOOGLE_API_KEY=your_google_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=webapp
MONGO_URI="mongodb+srv://<user>:<password>@cluster0.ibht45g.mongodb.net/?appName=Cluster0"
```

## 🏃‍♂️ Running the Application

### Local Development
To run the Flask development server on port `5000`:
```bash
python run.py
```
*The app will be accessible at `[http://127.0.0.1:5000/](http://127.0.0.1:5000/)`*.

### Production Deployment
For production, use the included Gunicorn WSGI server:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 run:app
```

## 📚 Data Ingestion (Vector Database)

To populate your Pinecone vector database with reference material, you can use the provided ingestion script.

1. Place your target `.docx` files in a local folder.
2. Open `scripts/ingest.py` and update the `DOCS_FOLDER` variable at the bottom of the script to point to your folder path.
3. Run the script:
```bash
python scripts/ingest.py
```
*This will chunk the text, generate embeddings, create the Pinecone index (if it doesn't exist), and upload the vectors in batches*.

## 📁 Project Structure

*   `app/__init__.py`: Flask application factory and CORS setup.
*   `app/routes.py`: Core API endpoints (`/api/chat`, `/api/history`).
*   `app/rag.py`: RAG chain initialization and retrieval logic.
*   `app/evaluator.py`: AI response safety and quality checking.
*   `app/mongo_history.py`: MongoDB interaction for saving/loading chat contexts.
*   `app/templates/index.html`: The front-end user interface.
*   `scripts/ingest.py`: Vector database population script.
*   `run.py`: Application entry point.
