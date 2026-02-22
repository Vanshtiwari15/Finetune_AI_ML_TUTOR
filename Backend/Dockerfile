FROM python:3.10-slim
LABEL maintainer="vanshtiwari1404"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# 1. Install CPU-only Torch first (Fast download, ~200MB vs 2GB)
# This makes the build much faster and "universal" (works on any cloud server without GPU)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest of the requirements
# We use --no-cache-dir and line 20-22 will copy the code later to leverage caching
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and backend code
COPY model/ /app/model/
COPY Backend/ /app/Backend/

WORKDIR /app/Backend

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]