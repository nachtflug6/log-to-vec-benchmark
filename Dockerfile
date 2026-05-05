FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Important: do not include torch / torchvision / torchaudio in requirements.txt,
# because the PyTorch CUDA image already includes GPU-enabled PyTorch.
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

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
