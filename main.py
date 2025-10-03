# api_rag_control.py

import os
import json
import threading
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

# LangChain / RAG imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration (Set your actual secrets here or via environment variables) ---
# ⚠️ Replace these placeholders with your actual secrets ⚠️
OPENAI_API_KEY =  os.getenv("OPENAI_API_KEY")
MONGO_CLIENT_KEY = os.getenv("MONGO_CLIENT_KEY")

# --- Constants ---
STATE_FILE = ".processed_docs.json"
DB_NAME = "vector_db"
COLLECTION_NAME = "embeddings"
INDEX_NAME = "vector_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL_NAME = "gpt-4o-mini"
DOCUMENT_FOLDERS = [
    "D:/Work/AI/Appointment Letters",
    "D:/Work/AI/Resignation Letters"
]
# Global status flag for the embedding process
EMBEDDING_STATUS = {"status": "IDLE", "last_run": "Never", "message": "Ready to start embedding."}


# ------------------ RAG PIPELINE (Query Logic - from installer.py) ------------------
class RAGPipeline:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        self.vectorstore = None
        self.qa_chain = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        try:
            client = MongoClient(MONGO_CLIENT_KEY)
            collection = client[DB_NAME][COLLECTION_NAME]
            
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

            self.vectorstore = MongoDBAtlasVectorSearch(
                embedding=embeddings,
                collection=collection,
                index_name=INDEX_NAME,
            )
            
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model=LLM_MODEL_NAME, temperature=0),
                retriever=self.retriever,
                return_source_documents=True,
            )
            print("RAG Query Pipeline initialized.")
        except Exception as e:
            print(f"Error initializing RAG Query Pipeline: {e}")

    def ask(self, query: str) -> dict:
        if not self.vectorstore:
            raise RuntimeError("RAG pipeline not initialized. Database error.")
        try:
            result = self.qa_chain.invoke({"query": query})
            
            source_documents = [
                {"source": doc.metadata.get('source'), "page_content_snippet": doc.page_content[:200] + "..."}
                for doc in result.get("source_documents", [])
            ]

            return {
                "answer": result["result"],
                "query": query,
                "sources": source_documents
            }
        except Exception as e:
            raise RuntimeError(f"Error querying RAG: {e}")

# ------------------ RAG SETUP / EMBEDDING LOGIC (from setup.py) ------------------

def load_processed_state():
    """Loads the state of previously processed documents."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_processed_state(state):
    """Saves the current state of processed documents."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)

def run_incremental_setup():
    """Core function to run the incremental embedding process."""
    global EMBEDDING_STATUS
    EMBEDDING_STATUS.update({"status": "RUNNING", "message": "Embedding process started..."})
    print("Starting incremental RAG setup...")

    try:
        # 1. MongoDB Connection
        client = MongoClient(MONGO_CLIENT_KEY)
        collection = client[DB_NAME][COLLECTION_NAME]

        # 2. State & File Discovery
        processed_state = load_processed_state()
        current_files = {}

        for folder_path in DOCUMENT_FOLDERS:
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    if file.endswith(".docx") and os.path.isfile(file_path):
                        last_modified = os.path.getmtime(file_path)
                        current_files[file_path] = last_modified
            else:
                print(f"Warning: Document folder not found: {folder_path}")

        # 3. Identify changes (Removed and Added/Updated)
        files_to_remove = [p for p in processed_state.keys() if p not in current_files]
        files_to_add_or_update = [
            p for p, mod in current_files.items() 
            if p not in processed_state or processed_state[p] < mod
        ]
        
        if not files_to_add_or_update and not files_to_remove:
            EMBEDDING_STATUS.update({
                "status": "COMPLETED", 
                "last_run": datetime.now().isoformat(),
                "message": "No changes detected. Vector store up to date."
            })
            print("No changes detected. Setup complete.")
            return

        # 4. Initialize Embeddings & Splitter
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        # 5. Handle removals
        if files_to_remove:
            for file_path in files_to_remove:
                result = collection.delete_many({"metadata.source": file_path})
                print(f"Removed {result.deleted_count} embeddings for: {os.path.basename(file_path)}")
                if file_path in processed_state:
                    del processed_state[file_path]

        # 6. Handle new and updated files
        if files_to_add_or_update:
            new_docs = []
            for file_path in files_to_add_or_update:
                if file_path in processed_state:
                    # Clear old embeddings before processing the updated file
                    collection.delete_many({"metadata.source": file_path})

                try:
                    loader = Docx2txtLoader(file_path)
                    documents = loader.load()
                    chunks = text_splitter.split_documents(documents)
                    new_docs.extend(chunks)
                    processed_state[file_path] = current_files[file_path]
                except Exception as e:
                    print(f"  > ERROR processing file {file_path}: {e}. Skipping.")
                    continue
            
            if new_docs:
                print(f"Inserting {len(new_docs)} new chunks...")
                MongoDBAtlasVectorSearch.from_documents(
                    documents=new_docs,
                    embedding=embeddings,
                    collection=collection,
                    index_name=INDEX_NAME
                )
                print("Successfully inserted new embeddings.")

        # 7. Save State & Final Status
        save_processed_state(processed_state)
        EMBEDDING_STATUS.update({
            "status": "COMPLETED", 
            "last_run": datetime.now().isoformat(),
            "message": "Embedding process finished successfully."
        })
        print("Incremental RAG setup complete.")

    except Exception as e:
        EMBEDDING_STATUS.update({
            "status": "FAILED", 
            "last_run": datetime.now().isoformat(),
            "message": f"Embedding process failed: {str(e)}"
        })
        print(f"FATAL EMBEDDING ERROR: {e}")


# ------------------ FastAPI Application ------------------
app = FastAPI(
    title="RAG Control API Service",
    description="API to query and trigger incremental setup for the RAG system.",
    version="2.0.0"
)

# Initialize the RAG query pipeline globally
rag_pipeline = RAGPipeline()


# Pydantic models for API documentation
class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    
class QueryResponse(BaseModel):
    answer: str
    query: str
    sources: list

class SetupResponse(BaseModel):
    status: str
    message: str
    last_run: str


@app.get("/health", tags=["System"])
async def health_check():
    """Simple health check to ensure the server is running."""
    db_status = "OK" if rag_pipeline.vectorstore else "ERROR"
    return {"status": "UP", "rag_db_status": db_status}


@app.post("/query", response_model=QueryResponse, tags=["RAG Query"])
async def process_query(request: QueryRequest):
    """Retrieves an answer from the RAG pipeline."""
    if not rag_pipeline.vectorstore:
        raise HTTPException(status_code=503, detail="RAG Service is unavailable.")
    try:
        result = rag_pipeline.ask(request.query)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/setup/start_embedding", tags=["RAG Setup"])
async def start_embedding(background_tasks: BackgroundTasks):
    """Triggers the incremental embedding process as a background task."""
    global EMBEDDING_STATUS
    if EMBEDDING_STATUS["status"] == "RUNNING":
        raise HTTPException(status_code=429, detail="Embedding process is already running.")

    # Add the setup function to be run in the background
    background_tasks.add_task(run_incremental_setup)

    EMBEDDING_STATUS.update({"status": "SCHEDULED", "message": "Embedding process scheduled."})
    return {"status": "SUCCESS", "message": "Embedding process started in background."}


@app.get("/setup/status", response_model=SetupResponse, tags=["RAG Setup"])
async def get_embedding_status():
    """Checks the status of the last or currently running embedding process."""
    return EMBEDDING_STATUS


if __name__ == "__main__":
    import uvicorn
    # To run this file, save it as 'api_rag_control.py' and use:
    # uvicorn api_rag_control:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)