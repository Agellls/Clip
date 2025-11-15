# syntax=docker/dockerfile:1
FROM python:3.11-slim

# metadata
LABEL maintainer="you@example.com"
ENV PYTHONUNBUFFERED=1 \
    # default port used by uvicorn
    PORT=8000

# install system deps (ffmpeg, build deps for some libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        git \
        build-essential \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# create app user
RUN groupadd -r app && useradd -r -g app -m -d /home/app app
WORKDIR /home/app

# copy only dependencies first for caching
COPY requirements.txt /home/app/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /home/app/requirements.txt

# copy app
COPY . /home/app
RUN chown -R app:app /home/app

USER app
EXPOSE 8000

# default command (adjust workers if needed)
CMD ["uvicorn", "clip_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
