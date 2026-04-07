FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/env
WORKDIR /app/env

RUN pip install --upgrade pip && \
    pip install "openenv-core[core]>=0.2.2" && \
    pip install .

ENV ENABLE_WEB_INTERFACE=true
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
