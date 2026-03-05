# ============================================================================
# Python ai-pipeline — Docker build
# Uses Python 3.11 with GPU-optional torch installation
# ============================================================================
FROM python:3.11-slim-bookworm

# Install system dependencies for spaCy and protobuf tooling
RUN apt-get update && apt-get install -y \
    build-essential \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first for layer caching
COPY services/ai-pipeline/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model for NER (transformer-based for best accuracy)
RUN python -m spacy download en_core_web_trf || python -m spacy download en_core_web_sm

# Copy proto files and generate Python gRPC stubs
COPY proto/ /app/proto/
RUN python -m grpc_tools.protoc \
    -I/app/proto \
    --python_out=/app/src/grpc \
    --grpc_python_out=/app/src/grpc \
    /app/proto/common.proto \
    /app/proto/graph.proto \
    /app/proto/ai.proto || true

# Copy application source code
COPY services/ai-pipeline/src/ ./src/
COPY services/ai-pipeline/tests/ ./tests/

# Expose gRPC port
EXPOSE 50052
# Expose FastAPI HTTP port (for health checks and optional REST API)
EXPOSE 8001

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8001/health')" || exit 1

CMD ["python", "-m", "src.main"]
