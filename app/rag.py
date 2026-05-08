import os
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.mongo_history import MongoHistoryManager
from app.evaluator import ResponseEvaluator

"""
Retrieval-Augmented Generation (RAG) Service Module.

This module initializes and orchestrates the RAG pipeline. It connects to Pinecone
for document retrieval, uses Google Generative AI for embeddings and response
generation, and integrates chat history and safety evaluations.
"""

logger = logging.getLogger(__name__)


class RAG:
    """
    A service class encapsulating the Retrieval-Augmented Generation pipeline.

    Attributes:
        embeddings (GoogleGenerativeAIEmbeddings): The embedding model used for queries.
        index_name (str): The name of the Pinecone index.
        vectorstore (PineconeVectorStore): The vector store interface.
        retriever (VectorStoreRetriever): The document retriever set to fetch top 5 results.
        llm (ChatGoogleGenerativeAI): The core language model for generation.
        rag_chain (Runnable): The compiled LangChain retrieval and generation chain.
        history_manager (MongoHistoryManager): The service managing MongoDB interactions.
        evaluator (ResponseEvaluator): The service evaluating AI responses for quality.
    """

    def __init__(self):
        logger.info("Initializing RAG Service...")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")
        self.index_name = os.environ.get("PINECONE_INDEX_NAME")
        self.vectorstore = PineconeVectorStore(index_name=self.index_name, embedding=self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        self.llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.2, max_retries = 3)
        self.rag_chain = self._build_chain()
        self.history_manager = MongoHistoryManager()
        self.evaluator = ResponseEvaluator()

    def _build_chain(self):
        """
        Construct the LangChain retrieval and question-answering chain.

        Combines a history-aware retriever (to contextualize follow-up questions)
        with a document-stuffing QA chain using the primary LLM.

        Returns:
            Runnable: The fully constructed RAG chain ready for invocation.
        """

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(self.llm, self.retriever, contextualize_q_prompt)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert full-stack developer assistant. Generate production-level code, necessary setup commands, and project summaries. Use the following retrieved context to answer the question. If you don't know the answer based on the context, use your deep programming knowledge to fill the gaps. Format code blocks beautifully.\n\nContext:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        return create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_response(self, session_id, user_input, file_path = None):
        """
        Generate and evaluate a response for a given user query.

        Retrieves the past 10 messages for context, invokes the RAG chain to
        generate a draft response, and passes the draft through the ResponseEvaluator.
        If the response passes evaluation, it is saved to MongoDB.

        Args:
            session_id (str): The unique identifier for the chat session.
            user_input (str): The user's prompt or question.
            file_path (str, optional): Path to an uploaded file (PDF or TXT) to provide
                                       temporary context for the query. Defaults to None.
        Returns:
            str: The final evaluated response or a safe fallback message.
        """

        chat_history = self.history_manager.get_history(session_id, limit=10)
        db_user_input = user_input
        if file_path:
            filename = os.path.basename(file_path)
            # This matches the nice UI format your frontend uses
            db_user_input = f"{user_input}<br><small><i>Attached: {filename}</i></small>"
            
        file_context = ""
        if file_path:
            try:
                if file_path.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path)

                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)

                file_context = "\n\n".join([chunk.page_content for chunk in chunks])
                logger.info(f"Successfully chunked {len(chunks)} tokens for temporary context.")
            except Exception as e:
                logger.error(f"Failed to process temporary file: {str(e)}")
                file_context = f"[Error reading attached file: {str(e)}]"

        if file_context:
            user_input = f"The user has attached a document with the following content:\n\n<document>\n{file_context}\n</document>\n\nUser Question:\n{user_input"

        response = self.rag_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })

        draft_response = response["answer"]

        passed, final_response = self.evaluator.evaluate(user_input, draft_response)

        if passed:
            self.history_manager.save_interaction(session_id, user_input, final_response, limit=10)

        return final_response

# Instantiate a single service for the app
rag = RAG()
