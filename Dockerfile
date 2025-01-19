FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /opt/ml/code

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    sagemaker-training \
    sagemaker-pytorch-training

# Set environment variables for GPU support
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Copy training code and dependencies
COPY . .

# Make training script executable
RUN chmod +x full_train.py

# Set entry point for SageMaker
ENTRYPOINT ["python3", "full_train.py"]