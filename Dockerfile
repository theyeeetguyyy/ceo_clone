# ══════════════════════════════════════════════════════════════
# CEO Digital Twin — Dockerfile for Hugging Face Spaces
# Port: 7860 (HF standard)
# ══════════════════════════════════════════════════════════════

FROM python:3.12-slim

# System deps needed by sentence-transformers / chromadb Rust core
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (HF Spaces security requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Upgrade pip first to avoid stale resolver issues
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies first (layer cache optimisation)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . .

# Ensure data directory skeleton exists (prevents crash if LFS files are stubs)
RUN mkdir -p data/vector_store data/memory_store logs

# Expose port 7860 (HF Spaces default)
EXPOSE 7860

# Healthcheck so HF Spaces can detect readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Run the FastAPI application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
