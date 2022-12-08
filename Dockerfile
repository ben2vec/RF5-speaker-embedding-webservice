# Modeled after https://github.com/ahmetoner/whisper-asr-webservice/blob/main/Dockerfile
FROM python:3.9.9-slim

# Install FFMPEG
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get -qq install --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Torch is big and annoying, install it first
RUN pip install torch==1.13.0
RUN pip install torchaudio==0.13.0

# So is librosa, install it next
RUN pip install librosa==0.9.2

# Download the model
COPY download_model.py .
RUN python download_model.py

# Install the webserver
RUN pip install gunicorn==20.1.0 uvicorn==0.18.2 "fastapi==0.88.*" "fastapi-offline==1.5.*"

# Install the rest of the deps
RUN pip install tqdm numpy pandas ffmpeg-python python-multipart 

# Copy the webservice files
WORKDIR /app
COPY . /app

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9000", "--workers", "1", "--timeout", "0", "app.webservice:app", "-k", "uvicorn.workers.UvicornWorker"]
