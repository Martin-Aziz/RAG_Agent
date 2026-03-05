"""
grpc/server.py — gRPC servicer implementing all AIService RPCs.

Maps protobuf requests to the Python pipeline components and streams
results back via gRPC server streaming.
"""

from __future__ import annotations

import grpc
from concurrent import futures
from typing import Optional

from loguru import logger

from src.models.schemas import (
    ChatMessage, ChatOptions, ChatRequest, ChatToken, TokenType,
    IngestProgress, IngestStage,
)
from src.nlp.embedder import SentenceEmbedder
from src.nlp.extractor import KnowledgeExtractor
from src.llm.client import LLMClient
from src.rag.pipeline import KnowledgeGraphRAGPipeline


class AIServiceServicer:
    """gRPC servicer implementing the AIService defined in ai.proto.

    Routes gRPC calls to the appropriate pipeline components:
    - IngestDocument → NLP extraction pipeline
    - Chat → RAG pipeline with LLM streaming
    - GenerateEmbedding → Sentence embedder
    - ExtractEntities → NER + relation extraction
    """

    def __init__(
        self,
        embedder: SentenceEmbedder,
        extractor: KnowledgeExtractor,
        llm_client: LLMClient,
        rag_pipeline: KnowledgeGraphRAGPipeline,
    ):
        """Initialize the servicer with all pipeline components."""
        self.embedder = embedder
        self.extractor = extractor
        self.llm = llm_client
        self.rag_pipeline = rag_pipeline

        logger.info("AIServiceServicer initialized")

    async def IngestDocument(self, request, context):
        """Process a document through the full NLP pipeline.

        Server-streaming RPC: yields IngestProgress updates for each stage.
        """
        logger.info(
            f"IngestDocument: doc_id={request.document_id}, "
            f"content_len={len(request.content)}"
        )

        try:
            async for progress in self.rag_pipeline.ingest_document(
                document_id=request.document_id,
                content=request.content,
                title=request.title,
                metadata=dict(request.metadata),
            ):
                # Convert internal model to protobuf response
                # In production, this uses the generated protobuf types
                yield progress

        except Exception as e:
            logger.exception(f"IngestDocument failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

    async def Chat(self, request, context):
        """Process a chat message through the RAG pipeline.

        Server-streaming RPC: yields ChatToken for each generated token.
        """
        logger.info(
            f"Chat: session={request.session_id}, "
            f"message='{request.message[:80]}...'"
        )

        try:
            # Convert protobuf history to internal models
            history = [
                ChatMessage(role=msg.role, content=msg.content)
                for msg in request.history
            ]

            options = ChatOptions(
                temperature=request.options.temperature if request.options else 0.7,
                max_tokens=request.options.max_tokens if request.options else 1024,
                include_graph_citation=True,
                stream_subgraph=True,
            )

            async for token in self.rag_pipeline.chat(
                message=request.message,
                history=history,
                session_id=request.session_id,
                options=options,
            ):
                yield token

        except Exception as e:
            logger.exception(f"Chat failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

    async def GenerateEmbedding(self, request, context):
        """Generate embeddings for input texts.

        Unary RPC: returns all embeddings in a single response.
        """
        logger.info(f"GenerateEmbedding: {len(request.texts)} texts")

        try:
            results = await self.embedder.embed_batch(list(request.texts))
            # Convert to protobuf response
            return results

        except Exception as e:
            logger.exception(f"GenerateEmbedding failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

    async def ExtractEntities(self, request, context):
        """Extract entities and relations from input text.

        Unary RPC: returns extracted entities and relations.
        """
        logger.info(f"ExtractEntities: text_len={len(request.text)}")

        try:
            entities, relations = self.extractor.extract(
                text=request.text,
                document_id=request.document_id,
            )

            return {
                "entities": entities,
                "relations": relations,
            }

        except Exception as e:
            logger.exception(f"ExtractEntities failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))


def create_grpc_server(
    embedder: SentenceEmbedder,
    extractor: KnowledgeExtractor,
    llm_client: LLMClient,
    rag_pipeline: KnowledgeGraphRAGPipeline,
    port: int = 50052,
) -> grpc.aio.Server:
    """Create and configure the gRPC server with the AI service.

    Args:
        embedder: Sentence embedding model.
        extractor: NER + relation extraction pipeline.
        llm_client: LLM inference client.
        rag_pipeline: RAG orchestrator.
        port: gRPC server port.

    Returns:
        Configured gRPC server (not yet started).
    """
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ],
    )

    servicer = AIServiceServicer(
        embedder=embedder,
        extractor=extractor,
        llm_client=llm_client,
        rag_pipeline=rag_pipeline,
    )

    # In production, this would register the generated service:
    # ai_pb2_grpc.add_AIServiceServicer_to_server(servicer, server)

    server.add_insecure_port(f"[::]:{port}")

    logger.info(f"gRPC server configured on port {port}")
    return server
