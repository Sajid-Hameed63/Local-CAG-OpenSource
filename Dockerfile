# Use a lightweight Python image
FROM python:3.10-slim

# Set environment variables to optimize Python and pip
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Install system dependencies (including curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy and install Python dependencies first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . /app/

# Expose Streamlit's default port
EXPOSE 8501

# Start Ollama service, pull models, and launch Streamlit (at runtime)
CMD ollama pull deepseek-r1:1.5b && ollama pull nomic-embed-text:latest && ollama serve & sleep 5 && streamlit run app.py --server.port=8501 --server.address=0.0.0.0

# sudo docker build -t local-cag-opensource:v1.0 .
# sudo docker run --env-file .env -p 8500:8500 local-cag-opensource:v1.0

