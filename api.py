from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
from materials_rag import MaterialsRAGSystem
from materials_data import sample_dataset
from sse_starlette.sse import EventSourceResponse
import asyncio
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    """
    Request model for query endpoints.
    
    Attributes:
        question (str): The user's question to be processed
        init_new_session (bool): Flag to start a new conversation session
    """
    question: str
    init_new_session: bool = False

class QueryResponse(BaseModel):
    """
    Response model for query endpoints.
    
    Attributes:
        answer (str): The generated answer to the user's question
        sources (List[str]): List of source documents used to generate the answer
    """
    answer: str
    sources: List[str]

class DocumentStats(BaseModel):
    """
    Model for vector store statistics.
    
    Attributes:
        total_vectors (int): Total number of vectors in the store
        persist_directory (str): Directory where vectors are stored
    """
    total_vectors: int
    persist_directory: str

class APIConfig:
    """
    Configuration settings for the API.
    
    Attributes:
        MODEL_PATH (str): Path to the language model
        CORS_ORIGINS (List[str]): Allowed origins for CORS
        CORS_METHODS (List[str]): Allowed HTTP methods
        CORS_HEADERS (List[str]): Allowed HTTP headers
    """
    MODEL_PATH = ""
    CORS_ORIGINS = ["*"]
    CORS_METHODS = ["*"]
    CORS_HEADERS = ["*"]

class MaterialsAPI:
    """
    Main API class for the Materials RAG system.
    
    This class handles all API endpoints and manages the interaction between
    the client and the RAG system. It provides functionality for querying
    materials information, managing chat sessions, and streaming responses.
    """
    
    def __init__(self):
        """
        Initialize the API with FastAPI app and RAG system.
        Sets up middleware and routes.
        """
        self.app = FastAPI(title="Materials RAG API")
        self.rag_system = MaterialsRAGSystem(APIConfig.MODEL_PATH, sample_dataset)
        self._setup_middleware()
        self._setup_routes()

    def _setup_middleware(self):
        """Configure CORS middleware for the API."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=APIConfig.CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=APIConfig.CORS_METHODS,
            allow_headers=APIConfig.CORS_HEADERS,
        )

    def _setup_routes(self):
        """Register all API routes with their respective handlers."""
        self.app.get("/")(self.root)
        self.app.post("/query", response_model=QueryResponse)(self.query_endpoint)
        self.app.post("/prepare-documents")(self.prepare_documents_endpoint)
        self.app.get("/health")(self.health_check)
        self.app.post("/query/stream")(self.query_stream_endpoint)
        self.app.post("/create_session")(self.create_session)
        self.app.get("/session/{session_id}")(self.get_session)

    async def root(self):
        """
        Root endpoint providing API information and available endpoints.
        
        Returns:
            dict: API information and endpoint listing
        """
        return {
            "message": "Welcome to Materials RAG API",
            "documentation": "/docs",
            "endpoints": {
                "query": "/query",
                "stream": "/query/stream",
                "stats": "/vector-store-stats",
                "health": "/health"
            }
        }

    async def query_endpoint(self, request: QueryRequest):
        """
        Process a question and return an answer.
        
        Args:
            request (QueryRequest): Contains the question and session preferences
            
        Returns:
            QueryResponse: The answer and source documents
            
        Raises:
            HTTPException: If query processing fails
        """
        try:
            response = self.rag_system.query(
                question=request.question,
                init_new_session=request.init_new_session
            )
            
            return QueryResponse(
                answer=response["answer"],
                sources=[str(doc) for doc in response.get("source_documents", [])]
            )
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def prepare_documents_endpoint(self, materials_data: Dict[str, Any]):
        """
        Prepare and index new materials documents.
        
        Args:
            materials_data (Dict[str, Any]): The materials data to be processed
            
        Returns:
            dict: Success status and message
            
        Raises:
            HTTPException: If document preparation fails
        """
        try:
            self.rag_system.prepare_documents(materials_data)
            return {"status": "success", "message": "Documents prepared successfully"}
        except Exception as e:
            logger.error(f"Error preparing documents: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def health_check(self):
        """
        Check API health status.
        
        Returns:
            dict: Health status indicator
        """
        return {"status": "healthy"}

    async def query_stream_endpoint(self, request: QueryRequest):
        """
        Stream the response for a question word by word.
        
        Args:
            request (QueryRequest): Contains the question and session preferences
            
        Returns:
            EventSourceResponse: Streaming response
        """
        async def generate():
            try:
                response_stream = self.rag_system.stream_query(
                    question=request.question,
                    init_new_session=request.init_new_session
                )
                
                async for chunk in response_stream:
                    if chunk is None:
                        continue
                    
                    yield self._format_sse_message("message", chunk)
                    
                yield self._format_sse_message("done", {"finish": True})
                
            except Exception as e:
                logger.error(f"Error in stream processing: {e}")
                yield self._format_sse_message("error", {"error": str(e)})
        
        return EventSourceResponse(generate())

    async def create_session(self):
        """
        Create a new chat session.
        
        Returns:
            dict: New session ID
            
        Raises:
            HTTPException: If session creation fails
        """
        try:
            session_id = self.rag_system.create_session()
            return {"session_id": session_id}
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_session(self, session_id: str):
        """
        Retrieve conversation history for a session.
        
        Args:
            session_id (str): ID of the session to retrieve
            
        Returns:
            dict: Session conversation history
            
        Raises:
            HTTPException: If session not found or retrieval fails
        """
        try:
            history = self.rag_system.get_session(session_id)
            if history is None:
                raise HTTPException(status_code=404, detail="Session not found")
            return {"history": history}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error retrieving session: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    def _format_sse_message(event: str, data: Any) -> dict:
        """
        Format a message for Server-Sent Events (SSE).
        
        Args:
            event (str): Event type identifier
            data (Any): Message data to be sent
            
        Returns:
            dict: Formatted SSE message
        """
        return {
            "event": event,
            "data": json.dumps(data)
        }

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application
    """
    api = MaterialsAPI()
    return api.app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
