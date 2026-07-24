# Use official PyTorch image matching CUDA 11.8 and PyTorch 2.5.1
FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

# Prevent interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch Geometric and its compiled dependencies matching torch 2.5.1 + cu118
RUN pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    pyg-lib \
    -f https://data.pyg.org/whl/torch-2.5.1+cu118.html

# Copy requirements file first to leverage docker caching
COPY requirements.txt /app/

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create runtime directories and a non-root user for executing research scripts
RUN mkdir -p /app/logs /app/checkpoints /app/final_result /app/data && \
    groupadd -g 1000 researcher && \
    useradd -u 1000 -g researcher -m researcher && \
    chown -R researcher:researcher /app

# Switch to the non-root user
USER researcher

# Copy the rest of the codebase
COPY --chown=researcher:researcher . /app

# Default command runs the test suite to verify integrity
CMD ["pytest", "tests/", "-v"]
