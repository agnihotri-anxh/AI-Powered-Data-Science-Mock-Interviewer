FROM python:3.11-slim

WORKDIR /app

# Install system dependencies with minimal footprint
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --retries 3 --timeout 300 -r requirements.txt

# Copy application code
COPY . .
RUN mkdir -p knowledge_base

# Expose port
EXPOSE 5000

# Environment variables for memory optimization
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Memory optimization settings
ENV MALLOC_TRIM_THRESHOLD_=131072
ENV MALLOC_MMAP_THRESHOLD_=131072
ENV MALLOC_MMAP_MAX_=65536
ENV MALLOC_ARENA_MAX=2

# Python memory optimization
ENV PYTHONHASHSEED=random
ENV PYTHONIOENCODING=utf-8

# Set memory limits for the container
ENV MEMORY_LIMIT=400m

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Use exec form to ensure proper signal handling
CMD ["python", "app.py"]
