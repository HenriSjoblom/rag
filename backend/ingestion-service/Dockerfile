FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY ./documents ./documents
COPY requirements.txt .

# Install system dependencies including libGL
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# Use cpu version of torch
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.7.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir onnxruntime==1.19.2 && \
    pip install --no-cache-dir -r requirements.txt

COPY ./app ./app

EXPOSE 8004

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
