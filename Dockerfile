# Use official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

RUN ollama pull deepseek-r1:1.5b

RUN ollama serve

# Expose Streamlit default port
EXPOSE 8500

# Run the Streamlit app (dotenv will be loaded at runtime)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# sudo docker build -t local-cag-opensource:v1.0 .
# docker run --env-file .env -p 8500:8500 local-cag-opensource:v1.0

