FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Create app directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories for results
RUN mkdir -p Experiment_Results/Final/Video Experiment_Results/Final/txt

# Default command
CMD ["python", "predict_drone.py", "--input", "Data/Videos/Clip_1.mov", "--output", "Experiment_Results/Final/Video/output.mp4"]
