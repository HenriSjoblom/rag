# RAG Backend System

A microservices-based Retrieval-Augmented Generation (RAG) system for processing user manuals and providing intelligent question answering. Upload technical documentation, product manuals, or user guides and get answers based on the manual content.

## Architecture

Four microservices orchestrated by Docker Compose:

![rag_diagram_en](https://github.com/user-attachments/assets/016190c1-4630-4a39-9fdf-788e791d836c)

### Service Responsibilities

#### RAG Service

- **Main orchestrator** that coordinates all other services
- Handles chat requests by calling retrieval → generation pipeline
- Manages document uploads by forwarding to ingestion service
- Provides unified API for frontend applications

#### Retrieval Service

- Performs **vector similarity search** on document embeddings
- Connects to ChromaDB for efficient retrieval
- Uses sentence transformers for query embeddings
- Returns relevant document chunks for context

#### Generation Service

- **LLM-powered response generation** using OpenAI API
- Takes user queries and retrieved context to generate answers

#### Ingestion Service

- **Document processing pipeline** for PDF files
- Extracts text, chunks documents, and creates embeddings
- Stores processed chunks in ChromaDB
- Manages document lifecycle (upload, process, delete)


## Technologies

**Backend Framework:**

- FastAPI - Web framework for building APIs
- Python 3.11+ - Programming language
- OpenAI GPT-4.1 - Large Language Model for text generation
- Sentence Transformers - Text embedding models
- ChromaDB - Vector database for similarity search
- Langchain - RAG pipeline orchestration framework

**Infrastructure:**

- Docker & Docker Compose - Containerization and orchestration

**RAG Pipeline:**

1. **Ingestion** - PyPDF extracts text, Langchain splits into chunks, Sentence Transformers create embeddings, ChromaDB stores vectors
2. **Retrieval** - Sentence Transformers embed user questions, ChromaDB performs vector similarity search to find relevant chunks
3. **Generation** - OpenAI GPT-4.1 receives question + retrieved context to generate answers

## Quick Start (Backend Only)

1. **Configure environment**

   ```bash
   # Copy and edit .env files
   cp rag_service/.env.example rag_service/.env
   cp retrieval_service/.env.example retrieval_service/.env
   cp generation_service/.env.example generation_service/.env
   cp ingestion-service/.env.example ingestion-service/.env

   # Add your OpenAI API key to generation_service/.env
   ```

2. **Start all services**

   ```bash
   docker-compose -f compose.backend.yml up -d
   ```

3. **Verify health**
   ```bash
   curl http://localhost:8001/health
   ```

## Quick Start with Frontend (Testing)

The frontend provides a web interface for testing the RAG system functionality.

1. **Configure environment**

   ```bash
   # Copy and edit backend .env files
   cp rag_service/.env.example rag_service/.env
   cp retrieval_service/.env.example retrieval_service/.env
   cp generation_service/.env.example generation_service/.env
   cp ingestion-service/.env.example ingestion-service/.env

   # Add your OpenAI API key to generation_service/.env
   ```

2. **Start backend and frontend**

   ```bash
   # Start all services including frontend
   docker-compose -f compose.yml up -d
   ```

3. **Access the testing interface**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8001/docs

## Key API Endpoints

### Upload & Chat (Main workflow)

```bash
# Upload document
curl -X POST http://localhost:8001/api/v1/documents/upload \
  -F "file=@document.pdf"

# Ask questions
curl -X POST http://localhost:8001/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What does the document say about X?"}'
```

## Configuration

Each service is configured via environment variables.

#### RAG Service (`rag_service/.env`)

| Variable                 | Required | Default                          | Description                    |
| :----------------------- | :------- | :------------------------------- | :----------------------------- |
| `RETRIEVAL_SERVICE_URL`  | Yes      | `http://retrieval_service:8002`  | URL to the retrieval service.  |
| `GENERATION_SERVICE_URL` | Yes      | `http://generation_service:8003` | URL to the generation service. |
| `INGESTION_SERVICE_URL`  | Yes      | `http://ingestion_service:8004`  | URL to the ingestion service.  |

#### Retrieval Service (`retrieval_service/.env`)

| Variable                 | Required | Default            | Description                                |
| :----------------------- | :------- | :----------------- | :----------------------------------------- |
| `CHROMA_MODE`            | Yes      | `docker`           | docker or local. Local only for testing    |
| `CHROMA_HOST`            | Yes      | `chromadb`         | Hostname of the ChromaDB service.          |
| `CHROMA_PORT`            | Yes      | `8000`             | Port for the ChromaDB service.             |
| `CHROMA_COLLECTION_NAME` | Yes      | `support_docs`     | ChromaDb collection name                   |
| `EMBEDDING_MODEL_NAME`   | Yes      | `all-MiniLM-L6-v2` | Sentence transformer model for embeddings. |
| `TOP_K_RESULTS`          | Yes      | `5`                | Number of relevant chunks to retrieve.     |

#### Generation Service (`generation_service/.env`)

| Variable         | Required | Default  | Description                                    |
| :--------------- | :------- | :------- | :--------------------------------------------- |
| `OPENAI_API_KEY` | **Yes**  | `""`     | **Your secret OpenAI API key.**                |
| `LLM_MODEL`      | Yes      | `gpt`    | OpenAI model to use for generation.            |
| `LLM_PROVIDER`   | Yes      | `openai` | LLM provider. Currently only Open AI available |

#### Ingestion Service (`ingestion-service/.env`)

| Variable                 | Required | Default            | Description                                                |
| :----------------------- | :------- | :----------------- | :--------------------------------------------------------- |
| `CHROMA_MODE`            | Yes      | `docker`           | docker or local. Local only for testing                    |
| `CHROMA_HOST`            | Yes      | `chromadb`         | Hostname of the ChromaDB service.                          |
| `CHROMA_PORT`            | Yes      | `8000`             | Port for the ChromaDB service.                             |
| `SOURCE_DIRECTORY`       | Yes      | `/app/documents`   | Directory inside the container for storing uploaded files. |
| `EMBEDDING_MODEL_NAME`   | Yes      | `all-MiniLM-L6-v2` | Sentence transformer model for embeddings.                 |
| `CHROMA_COLLECTION_NAME` | Yes      | `support_docs`     | ChromaDB collection name                                   |

## API Documentation

Interactive docs available at:

- http://localhost:8001/docs (RAG Service)
- http://localhost:8002/docs (Retrieval)
- http://localhost:8003/docs (Generation)
- http://localhost:8004/docs (Ingestion)

## Troubleshooting

**Services won't start:**

```bash
docker-compose -f compose.backend.yml logs [service-name]
```

**ChromaDB connection failed:**

```bash
curl http://localhost:8000/api/v1/heartbeat
```

**OpenAI API errors:**

- Check API key in `generation_service/.env`
- Verify account has credits
- Monitor rate limits in logs

### Reset Everything

```bash
docker-compose -f compose.backend.yml down
docker volume rm backend_chroma_data
docker-compose -f compose.backend.yml up -d
```
