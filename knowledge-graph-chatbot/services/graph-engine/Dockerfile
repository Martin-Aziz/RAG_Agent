# ============================================================================
# Rust graph-engine — Multi-stage Docker build
# Stage 1: Build with full Rust toolchain
# Stage 2: Slim Debian runtime with only the compiled binary
# ============================================================================
FROM rust:1.75-bookworm AS builder

# Install protobuf compiler for tonic-build
RUN apt-get update && apt-get install -y protobuf-compiler libclang-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy proto definitions first (they rarely change, maximizing layer cache)
COPY proto/ /app/proto/

# Copy Cargo files for dependency caching
COPY services/graph-engine/Cargo.toml services/graph-engine/build.rs ./

# Create a dummy main.rs to build dependencies first (Docker layer caching)
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release || true
RUN rm -rf src

# Copy actual source code
COPY services/graph-engine/src/ ./src/

# Build the final binary
RUN cargo build --release

# ============================================================================
# Runtime stage — minimal image with just the binary and required libs
# ============================================================================
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the compiled binary
COPY --from=builder /app/target/release/graph-engine /app/graph-engine

# Create data directories for RocksDB and HNSW index persistence
RUN mkdir -p /data/graph /data/vectors

# Expose gRPC port
EXPOSE 50051

# Health check via gRPC reflection (optional, can also use TCP check)
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD ["/app/graph-engine", "--health-check"] || exit 1

ENTRYPOINT ["/app/graph-engine"]
