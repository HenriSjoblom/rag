FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Create a non-root user and group
RUN addgroup --system app && adduser --system --ingroup app app
# Create data directory and set permissions if CHROMA_MODE=local was used (not for http mode to chromadb service)
RUN mkdir -p /app/data/chroma_db && chown -R app:app /app/data

COPY requirements.txt .
# Use cpu version of torch
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.7.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Switch to non-root user before copying app code
# USER app

COPY ./app ./app

# Ensure the data directory for ChromaDB (if local mode was used) has correct permissions
# This is less relevant if connecting to a ChromaDB service via HTTP.
# If CHROMA_LOCAL_PATH was relative like './data/chroma_db', it would be /app/data/chroma_db here.
RUN mkdir -p /app/data/chroma_db && chown -R app:app /app/data/chroma_db

EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
