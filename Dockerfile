FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV PATH="/opt/conda/bin:${PATH}"
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y ffmpeg && \
    apt-get install -y libsm6 && \
    apt-get install -y libxext6 && \
    apt-get install -y gcc && \
    apt-get install -y g++ && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    opencv-python-headless==4.8.1.78 \
    boto3==1.34.14 \
    sagemaker-training \
    smdebug

COPY requirements.txt /opt/ml/code/requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt -f https://download.pytorch.org/whl/cu121/torch_stable.html

COPY sm_train.py /opt/ml/code/train.py
COPY full_scale_classifier.py /opt/ml/code/model.py
COPY dataset.py /opt/ml/code/dataset.py

ENV SAGEMAKER_PROGRAM train.py
WORKDIR /opt/ml/code