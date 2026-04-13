

FROM python:3.11-slim

# Install OS dependencies for document parsing and Azure SDKs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt1-dev \
        libffi-dev \
        libssl-dev \
        libpq-dev \
        gcc \
        git \
        unzip \
        && rm -rf /var/lib/apt/lists/*

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app
WORKDIR /app

# Set required environment variables for Azure/OpenAI (documented for deployment)
ENV AZURE_SEARCH_ENDPOINT="" \
    AZURE_SEARCH_API_KEY="" \
    AZURE_SEARCH_INDEX_NAME="" \
    AZURE_OPENAI_ENDPOINT="" \
    AZURE_OPENAI_API_KEY="" \
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT=""

# Expose API port
EXPOSE 8000

# Startup command
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8000"]

