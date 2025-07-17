# RAG System for User Manual Assistant

A microservices-based Retrieval-Augmented Generation (RAG) system designed to answer questions about user manuals. It includes a complete backend that orchestrates document processing and response generation, along with a React frontend for testing.

## Architecture Overview

The RAG backend consists of four main microservices:

![rag_diagram_en](https://github.com/user-attachments/assets/016190c1-4630-4a39-9fdf-788e791d836c)

### Technology Stack

| Component | Technology/Library |
| :--- | :--- |
| Backend Framework | FastAPI |
| Vector Database | ChromaDB |
| LLM Integration | OpenAI API |
| Embedding Model | Sentence Transformers |
| Containerization | Docker & Docker Compose |

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


## Configuration

## Configuration

Each service is configured via environment variables.

#### RAG Service (`rag_service/.env`)
| Variable | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `RETRIEVAL_SERVICE_URL` | Yes | `http://retrieval_service:8002` | URL to the retrieval service. |
| `GENERATION_SERVICE_URL`| Yes | `http://generation_service:8003` | URL to the generation service. |
| `INGESTION_SERVICE_URL` | Yes | `http://ingestion_service:8004` | URL to the ingestion service. |

#### Retrieval Service (`retrieval_service/.env`)
| Variable | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `CHROMA_MODE` | Yes | `docker` | docker or local. Local only for testing |
| `CHROMA_HOST` | Yes | `chromadb` | Hostname of the ChromaDB service. |
| `CHROMA_PORT` | Yes | `8000` | Port for the ChromaDB service. |
| `CHROMA_COLLECTION_NAME`| Yes | `support_docs` | ChromaDb collection name |
| `EMBEDDING_MODEL_NAME`| Yes | `all-MiniLM-L6-v2` | Sentence transformer model for embeddings. |
| `TOP_K_RESULTS`| Yes | `5` | Number of relevant chunks to retrieve. |

#### Generation Service (`generation_service/.env`)
| Variable | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `OPENAI_API_KEY` | **Yes** | `""` | **Your secret OpenAI API key.** |
| `LLM_MODEL` | Yes | `gpt` | OpenAI model to use for generation. |
| `LLM_PROVIDER` | Yes | `openai` | LLM provider. Currently only Open AI available |

#### Ingestion Service (`ingestion-service/.env`)
| Variable | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `CHROMA_MODE` | Yes | `docker` | docker or local. Local only for testing |
| `CHROMA_HOST` | Yes | `chromadb` | Hostname of the ChromaDB service. |
| `CHROMA_PORT` | Yes | `8000` | Port for the ChromaDB service. |
| `SOURCE_DIRECTORY` | Yes | `/app/documents` | Directory inside the container for storing uploaded files. |
| `EMBEDDING_MODEL_NAME`| Yes | `all-MiniLM-L6-v2` | Sentence transformer model for embeddings. |
| `CHROMA_COLLECTION_NAME`| Yes | `support_docs` | ChromaDB collection name |


## Usage Examples

### Document Processing Workflow

1. **Upload a document**

   ```bash
   curl -X POST http://localhost:8001/api/v1/documents/upload \
     -F "file=@path/to/document.pdf"
   ```

2. **Ask questions about the document**
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

For more detailed information about individual services, check the service-specific documentation in each service directory.
