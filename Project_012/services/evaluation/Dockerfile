FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (curl for healthcheck, gcc/build-essential for compiling some Python packages)
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    gcc \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Silence GitPython warnings
ENV GIT_PYTHON_REFRESH=quiet

# Copy application code
COPY . .

EXPOSE 8001

# Healthcheck for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Entrypoint
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8101", "--loop", "asyncio", "--log-level", "debug"]