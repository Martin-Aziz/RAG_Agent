# ============================================================================
# Go api-gateway — Multi-stage Docker build
# ============================================================================
FROM golang:1.21-bookworm AS builder

# Install protobuf compiler
RUN apt-get update && apt-get install -y protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy go.mod and go.sum for dependency caching
COPY services/api-gateway/go.mod services/api-gateway/go.sum* ./
RUN go mod download

# Copy proto files and generate Go code
COPY proto/ /app/proto/

# Copy source code
COPY services/api-gateway/ ./

# Build the binary with optimizations
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /gateway ./cmd/server/

# ============================================================================
# Runtime — minimal scratch image
# ============================================================================
FROM gcr.io/distroless/static-debian12

WORKDIR /app
COPY --from=builder /gateway /app/gateway

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD ["/app/gateway", "--health-check"] || exit 1

ENTRYPOINT ["/app/gateway"]
