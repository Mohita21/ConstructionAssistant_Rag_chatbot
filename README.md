# Materials RAG (Retrieval-Augmented Generation) System

## Quick Start
To run the entire project with one command:

## Overview
The Materials RAG System is an intelligent conversational AI platform designed specifically for construction materials information retrieval and consultation. It combines advanced language models with a specialized knowledge base to provide accurate, context-aware responses about construction materials, specifications, building codes, and related documentation.

## Features

### 1. Intelligent Query Processing
- Advanced natural language understanding
- Context-aware responses
- Support for technical specifications and calculations
- Real-time streaming responses

### 2. Document Management
- Processes multiple document types:
  - Product catalogs
  - Technical documentation
  - Building codes
  - Installation guides
  - Safety documents
  - Material alternatives
- Automatic document indexing and vectorization
- Efficient retrieval system

### 3. Conversation Management
- Session-based conversations
- Conversation history tracking
- Multi-user support
- Stateful interactions

### 4. API Integration
- RESTful API endpoints
- Server-Sent Events (SSE) for streaming
- CORS support
- Health monitoring

## Technical Architecture

### Backend Components
1. **RAG System Core (`materials_rag.py`)**
   - Document processing and vectorization
   - LLM integration
   - Conversation management
   - Query processing

2. **API Layer (`api.py`)**
   - FastAPI implementation
   - Endpoint management
   - Request/response handling
   - Error management

3. **Configuration (`config.py`)**
   - Environment variables
   - API keys
   - System settings

### Frontend Components
- React-based user interface
- Real-time response streaming
- Markdown rendering
- Interactive chat interface

### Frontend Components (`materials-rag-chat/`)
The frontend is built using React and is organized in the `materials-rag-chat` directory:

1. **Core Components**
   - `src/components/`
     - `ChatInterface.js`: Main chat interface component
     - `MessageList.js`: Displays chat messages
     - `MessageInput.js`: User input component
     - `ResponseStream.js`: Handles streaming responses

2. **Styling and Layout**
   - `src/styles/`
     - `App.css`: Main application styles
     - `Chat.css`: Chat interface styling
     - `Messages.css`: Message components styling

3. **API Integration**
   - `src/services/`
     - `api.js`: API client for backend communication
     - `streamService.js`: Handles SSE streaming

4. **State Management**
   - `src/context/`
     - `ChatContext.js`: Chat state management
     - `SessionContext.js`: Session management

5. **Public Assets**
   - `public/`
     - `index.html`: Main HTML template
     - `assets/`: Images and icons
     - `fonts/`: Custom fonts

6. **Configuration**
   - `package.json`: Project dependencies
   - `.env`: Environment configuration
   - `config/`: Build and environment configs

## Project Files

### 1. Requirements File (`requirements.txt`)
Contains all necessary Python packages:
- FastAPI and server components
- LangChain and AI components
- Vector store and embeddings
- API utilities and middleware
- Production server

### 2. Start Script (`start_app.sh`)
Automated script that:
- Checks system prerequisites
- Installs Node.js if missing
- Manages port availability
- Starts backend and frontend servers
- Provides graceful shutdown

## Installation Guide

### Prerequisites
- Python 3.8 or higher
- Node.js 14 or higher (will be installed by start script if missing)
- OpenAI API key

### Detailed Setup Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/materials-rag.git
   cd materials-rag
   ```

2. **Set Up Python Virtual Environment**
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On Unix or MacOS:
   source venv/bin/activate

   # Verify activation (should show venv path)
   which python
   ```

3. **Install Python Dependencies**
   ```bash
   # Install all required packages from requirements.txt
   pip install -r requirements.txt

   # Verify installations
   pip list
   ```

4. **Configure Environment Variables**
   ```bash
   # Create .env file
   touch .env

   # Add your OpenAI API key
   echo "OPENAI_API_KEY=your_api_key_here" >> .env
   ```

5. **Launch the Application**

   **Option A: Using Start Script (Recommended)**
   ```bash
   # Make script executable (Unix/MacOS only)
   chmod +x start_app.sh

   # Run the start script
   ./start_app.sh
   ```

   **Option B: Manual Launch**
   ```bash
   # Terminal 1 - Start Backend
   python api.py

   # Terminal 2 - Start Frontend
   cd materials-rag-chat
   npm install
   npm start
   ```

6. **Access the Application**
   - Frontend Interface: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

7. **Stopping the Application**
   - If using start script: Press Ctrl+C (script handles cleanup)
   - If manual launch: Close both terminal windows (Ctrl+C in each)
