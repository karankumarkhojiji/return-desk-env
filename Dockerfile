# ============================================================
# ReturnDeskEnv — Dockerfile
# ============================================================
# Base image: Python 3.11 slim (keeps image size small)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
# curl  : needed for HEALTHCHECK
# git   : needed by some openenv dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the entire project into the container
COPY . /app

# Install the project and all its dependencies
# This reads pyproject.toml and installs everything listed there
RUN pip install --upgrade pip && \
    pip install "openenv-core[core]>=0.2.2" && \
    pip install .

# Tell Docker which port the app listens on
# (This is documentation only — does not actually open the port)
EXPOSE 8000

# Health check — HuggingFace uses this to know the Space is "Running"
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the server when the container runs
# --host 0.0.0.0 = accept connections from outside the container (required!)
# --port 8000    = must match EXPOSE above and openenv.yaml port
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
