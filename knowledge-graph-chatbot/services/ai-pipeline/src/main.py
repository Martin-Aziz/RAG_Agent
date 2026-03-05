"""
main.py — AI Pipeline service entrypoint.

Co-serves both FastAPI (HTTP for health checks) and gRPC (for service communication)
in a single process using asyncio.

The FastAPI app provides:
- /health — health check endpoint
- /docs — auto-generated API documentation

The gRPC server provides:
- AIService RPCs (IngestDocument, Chat, GenerateEmbedding, ExtractEntities)
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.nlp.embedder import SentenceEmbedder
from src.nlp.extractor import KnowledgeExtractor
from src.llm.client import LLMClient
from src.rag.pipeline import KnowledgeGraphRAGPipeline
from src.rag.prompt_builder import PromptBuilder
from src.rag.retriever import GraphAwareRetriever
from src.grpc.clients import GraphEngineClient
from src.grpc.server import create_grpc_server


# ============================================================================
# Configuration from environment variables
# ============================================================================

GRPC_PORT = int(os.getenv("GRPC_PORT", "50052"))
HTTP_PORT = int(os.getenv("HTTP_PORT", "8001"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
GRAPH_ENGINE_ADDR = os.getenv("GRAPH_ENGINE_ADDR", "graph-engine:50051")


# ============================================================================
# FastAPI app — HTTP endpoints for health checks and optional REST API
# ============================================================================

app = FastAPI(
    title="AI Pipeline",
    description="NLP extraction, embedding, and RAG inference service",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint for container orchestration."""
    return {
        "status": "ok",
        "service": "ai-pipeline",
        "grpc_port": GRPC_PORT,
    }


@app.get("/ready")
async def readiness():
    """Readiness probe — checks if models are loaded and service is ready."""
    return {
        "status": "ready",
        "models": {
            "embedder": "loaded",
            "extractor": "lazy",  # Loaded on first use
            "llm": "configured",
        },
    }


# ============================================================================
# Service initialization and startup
# ============================================================================

async def start_services():
    """Initialize all pipeline components and start both servers."""

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level=LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "{message}",
        serialize=False,
    )
    logger.add(
        "logs/ai-pipeline.log",
        rotation="100 MB",
        retention="7 days",
        level="DEBUG",
        serialize=True,
    )

    logger.info("=" * 60)
    logger.info("AI Pipeline service starting")
    logger.info("=" * 60)

    # ========================================================================
    # 1. Initialize pipeline components
    # ========================================================================

    # Sentence embedder (loads model lazily on first use)
    embedder = SentenceEmbedder()

    # Knowledge extractor (loads spaCy + REBEL lazily)
    extractor = KnowledgeExtractor()

    # LLM client (connects to Ollama or falls back to OpenAI)
    llm_client = LLMClient()

    # Graph-engine gRPC client
    graph_client = GraphEngineClient(addr=GRAPH_ENGINE_ADDR)
    try:
        await graph_client.connect()
        logger.info("Connected to graph-engine")
    except Exception as e:
        logger.warning(f"Could not connect to graph-engine: {e}. Running standalone.")
        graph_client = None

    # Prompt builder
    prompt_builder = PromptBuilder()

    # Graph-aware retriever
    retriever = GraphAwareRetriever(
        embedder=embedder,
        graph_client=graph_client,
    )

    # RAG pipeline orchestrator
    rag_pipeline = KnowledgeGraphRAGPipeline(
        embedder=embedder,
        llm_client=llm_client,
        graph_client=graph_client,
        prompt_builder=prompt_builder,
        retriever=retriever,
    )

    # ========================================================================
    # 2. Create gRPC server
    # ========================================================================
    grpc_server = create_grpc_server(
        embedder=embedder,
        extractor=extractor,
        llm_client=llm_client,
        rag_pipeline=rag_pipeline,
        port=GRPC_PORT,
    )

    # ========================================================================
    # 3. Start both servers concurrently
    # ========================================================================

    # Start gRPC server
    await grpc_server.start()
    logger.info(f"gRPC server started on port {GRPC_PORT}")

    # Start FastAPI server in background
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=HTTP_PORT,
        log_level="info",
    )
    http_server = uvicorn.Server(config)

    # Handle graceful shutdown
    async def shutdown(sig):
        logger.info(f"Received signal {sig}. Shutting down...")
        await grpc_server.stop(grace=5)
        if llm_client:
            await llm_client.close()
        if graph_client:
            await graph_client.close()
        logger.info("AI Pipeline shut down gracefully")

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig, lambda s=sig: asyncio.create_task(shutdown(s))
        )

    logger.info(f"HTTP server starting on port {HTTP_PORT}")

    # Run HTTP server (this blocks until shutdown)
    await http_server.serve()


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    asyncio.run(start_services())
