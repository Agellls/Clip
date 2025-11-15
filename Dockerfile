FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=8000

# install runtime + build deps commonly needed for building wheels
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    git \
    build-essential \
    python3-dev \
    pkg-config \
    curl \
    libsndfile1 \
    libsndfile1-dev \
    libffi-dev \
    libssl-dev \
    gcc \
    g++ \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /home/app
COPY requirements.txt /home/app/requirements.txt

# split steps so build log shows exact failing package
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip --version

RUN python -m pip install --no-cache-dir -r /home/app/requirements.txt
