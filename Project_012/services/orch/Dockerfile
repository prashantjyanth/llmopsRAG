FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (optional, useful for debugging and health checks)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y build-essential gcc
RUN pip install --upgrade pip
# Install Python dependencies first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENV GIT_PYTHON_REFRESH=quiet

# Copy the rest of the application code
COPY . .

EXPOSE 8001

# Healthcheck (optional, for Docker Compose)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]