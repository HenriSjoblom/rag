# RAG System for User Manual Assistant

A microservices-based Retrieval-Augmented Generation (RAG) system for document processing and question answering. This backend orchestrates multiple services to provide intelligent responses based on uploaded user manual documents.

## Architecture Overview

The RAG backend consists of four main microservices:

![rag_diagram_en](https://github.com/user-attachments/assets/016190c1-4630-4a39-9fdf-788e791d836c)

### Service Responsibilities

#### RAG Service (Port 8001)

- **Main orchestrator** that coordinates all other services
- Handles chat requests by calling retrieval → generation pipeline
- Manages document uploads by forwarding to ingestion service
- Provides unified API for frontend applications

#### Retrieval Service (Port 8002)

- Performs **vector similarity search** on document embeddings
- Connects to ChromaDB for efficient retrieval
- Uses sentence transformers for query embeddings
- Returns relevant document chunks for context

#### Generation Service (Port 8003)

- **LLM-powered response generation** using OpenAI API
- Takes user queries and retrieved context to generate answers
- Implements RAG prompt templates for better responses
- Handles LLM error cases and rate limiting

#### Ingestion Service (Port 8004)

- **Document processing pipeline** for PDF files
- Extracts text, chunks documents, and creates embeddings
- Stores processed chunks in ChromaDB
- Manages document lifecycle (upload, process, delete)

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key (for generation service)

### Using Docker Compose (Recommended)

1. **Clone and navigate to the backend directory**

   ```bash
   cd c:\Users\henri\Documents\projektit\rag\backend
   ```

2. **Configure environment variables**

   ```bash
   # Copy example env files
   cp rag_service/.env.example rag_service/.env
   cp retrieval_service/.env.example retrieval_service/.env
   cp generation_service/.env.example generation_service/.env
   cp ingestion-service/.env.example ingestion-service/.env

   # Edit generation_service/.env to add your OpenAI API key
   # OPENAI_API_KEY=your-api-key-here
   ```

3. **Start all services**

   ```bash
   docker-compose -f compose.backend.yml up -d
   ```

4. **Verify services are running**
   ```bash
   # Check service health
   curl http://localhost:8001/health  # RAG Service
   curl http://localhost:8002/health  # Retrieval Service
   curl http://localhost:8003/health  # Generation Service
   curl http://localhost:8004/health  # Ingestion Service
   curl http://localhost:8000/api/v1/heartbeat  # ChromaDB
   ```

## API Documentation

### RAG Service (Main API)

#### Health Check

```http
GET /health
```

#### Chat with Documents

```http
POST /api/v1/chat
Content-Type: application/json

{
    "message": "How do I reset my iPhone?"
}
```

#### Upload Documents

```http
POST /api/v1/documents/upload
Content-Type: multipart/form-data

file: [PDF file]
```

#### List Documents

```http
GET /api/v1/documents
```

#### Delete All Documents

```http
DELETE /api/v1/documents
```

#### Get Ingestion Status

```http
GET /api/v1/ingestion/status
```

### Direct Service APIs

#### Retrieval Service

```http
POST /api/v1/retrieve
Content-Type: application/json

{
    "query": "iPhone camera features"
}
```

#### Generation Service

```http
POST /api/v1/generate
Content-Type: application/json

{
    "query": "How do I use the camera?",
    "context_chunks": ["Camera section from manual..."]
}
```

#### Ingestion Service

```http
# Upload file
POST /api/v1/upload
Content-Type: multipart/form-data

# Trigger ingestion
POST /api/v1/ingest

# Check status
GET /api/v1/status

# Clear data
DELETE /api/v1/collection
```

## Development Setup

### Individual Service Development

1. **Set up Python environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies for a service**

   ```bash
   cd rag_service  # or retrieval_service, generation_service, ingestion-service
   pip install -r requirements.txt
   ```

3. **Run service locally**
   ```bash
   uvicorn app.main:app --reload --port 8001
   ```

### Testing

#### Run Unit Tests

```bash
# In each service directory
pytest tests/unit/ -v
```

#### Run Integration Tests

```bash
# In each service directory
pytest tests/integration/ -v
```



## Configuration

### Environment Variables

#### RAG Service

- `RETRIEVAL_SERVICE_URL`: URL to retrieval service
- `GENERATION_SERVICE_URL`: URL to generation service
- `INGESTION_SERVICE_URL`: URL to ingestion service

#### Retrieval Service

- `CHROMA_HOST`: ChromaDB host
- `CHROMA_PORT`: ChromaDB port
- `EMBEDDING_MODEL_NAME`: Sentence transformer model

#### Generation Service

- `OPENAI_API_KEY`: OpenAI API key (required)
- `LLM_MODEL_NAME`: OpenAI model to use
- `LLM_TEMPERATURE`: Response creativity (0.0-1.0)

#### Ingestion Service

- `CHROMA_HOST`: ChromaDB host
- `CHROMA_PORT`: ChromaDB port
- `SOURCE_DIRECTORY`: Directory for uploaded files

## Usage Examples

### Document Processing Workflow

1. **Upload a document**

   ```bash
   curl -X POST http://localhost:8001/api/v1/documents/upload \
     -F "file=@path/to/document.pdf"
   ```

2. **Wait for processing** (automatic with file upload)

   ```bash
   curl http://localhost:8001/api/v1/ingestion/status
   ```

3. **Ask questions about the document**
   ```bash
   curl -X POST http://localhost:8001/api/v1/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "What are the main features described in the document?"}'
   ```

## API Reference

### Complete OpenAPI Documentation

- RAG Service: http://localhost:8001/docs
- Retrieval Service: http://localhost:8002/docs
- Generation Service: http://localhost:8003/docs
- Ingestion Service: http://localhost:8004/docs

### Response Formats

#### Chat Response

```json
{
  "query": "How do I reset my device?",
  "response": "To reset your device, follow these steps: 1. Go to Settings..."
}
```

#### Document Upload Response

```json
{
  "status": "Upload accepted",
  "message": "File uploaded and ingestion started",
  "filename": "manual.pdf"
}
```

#### Retrieval Response

```json
{
  "chunks": [
    "Text chunk 1 from documents...",
    "Text chunk 2 from documents..."
  ],
  "collection_name": "documents",
  "query": "original query"
}
```

---

For more detailed information about individual services, check the service-specific documentation in each service directory.
