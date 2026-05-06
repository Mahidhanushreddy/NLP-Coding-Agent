import os
import logging
import time

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv
from pinecone import ServerlessSpec, Pinecone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

"""
Document Ingestion Pipeline Module.

This script scans a specified local directory for Word documents (.docx), extracts their text, 
splits the text into manageable chunks, generates vector embeddings using Google Generative AI, 
and uploads the vectors to a Pinecone vector database.
"""

def ingest_word_documents(folder_path: str):
    """
    Reads Word documents from a folder, chunks the text, embeds it using
    Gemini, and stores the vectors in Pinecone.

    Args:
        folder_path (str): The local system path to the directory containing .docx files.

    Returns:
        PineconeVectorStore: The initialized vector store instance after successful upload,
                             or None if the folder is invalid or contains no valid documents.
    """

    logger.info(f"Scanning folder '{folder_path}' for Word documents...")

    # 1. Read and Extract Data from Word Docs
    documents = []
    if not os.path.exists(folder_path):
        logger.error(f"The folder path '{folder_path}' does not exist.")
        return

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".docx"):
            file_path = os.path.join(folder_path, filename)
            logger.info(f"Loading document: {filename}")

            # Load the Word document
            loader = Docx2txtLoader(file_path)
            documents.append(loader.load())

    if not documents:
        logger.warning("No .docx files found in the specified directory.")
        return

    logger.info(f"Successfully loaded {len(documents)} document(s).")

    # 2. Chunk Data via Recursive Splitter
    # Utilizing the same chunking strategy as the RAG service

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []
    for document in documents:
        chunks.extend(text_splitter.split_documents(document))

    logger.info(f"Split documents into {len(chunks)} recursive chunks.")

    # 3. Initialize Gemini Embeddings
    # Utilizing the gemini-embedding-2-preview model from your existing configuration
    logger.info("Initializing Google Generative AI Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2-preview"
    )

    # 4. Store in Pinecone VectorDB
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")

    if not index_name:
        logger.error("PINECONE_INDEX_NAME environment variable is not set.")
        return
    if not pinecone_api_key:
        logger.error("PINECONE_API_KEY environment variable is not set.")
        return

    logger.info(f"Checking if Pinecone index '{index_name}' exists...")
    pc = Pinecone(api_key=pinecone_api_key)

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        logger.info(f"Index '{index_name}' does not exist. Creating it now...")
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # Update this region to match your Pinecone project settings
            )
        )

        # Wait for the index to be initialized before uploading vectors
        while not pc.describe_index(index_name).status['ready']:
            logger.info("Waiting for index to be ready...")
            time.sleep(2)

        logger.info(f"Index '{index_name}' successfully created.")
    else:
        logger.info(f"Index '{index_name}' already exists.")

    logger.info(f"Uploading vectors to Pinecone index: {index_name}...")

    # This automatically embeds the chunks and uploads them to the specified index
    vectorstore = PineconeVectorStore(
        embedding=embeddings,
        index_name=index_name
    )
    logger.info(f"Uploading vectors to Pinecone index: {index_name} in batches...")

    batch_size = 1  # Adjust based on your API rate limits
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        try:
            # add_documents handles the embedding and upload for the entire batch
            vectorstore.add_documents(batch)
            logger.info(f"Successfully uploaded batch {i // batch_size + 1} ({len(batch)} chunks).")

            time.sleep(1)
        except Exception as e:
            logger.error(f"Failed to upload batch starting at index {i}: {e}")
    logger.info("Ingestion complete! Vectors successfully stored in Pinecone.")
    return vectorstore


if __name__ == "__main__":
    # Specify the path to your folder containing the .docx files
    DOCS_FOLDER = r"C:\Users\DHANUSH\Downloads\Doc"

    # Execute the pipeline
    ingest_word_documents(DOCS_FOLDER)