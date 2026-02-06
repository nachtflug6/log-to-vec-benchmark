FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create data directory
RUN mkdir -p /app/data /app/checkpoints /app/outputs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Default command
CMD ["/bin/bash"]
