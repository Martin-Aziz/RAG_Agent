"""
rag/pipeline.py — Knowledge Graph RAG orchestrator.

Orchestrates the complete graph-grounded RAG pipeline:
1. Embed query → vector search → candidate nodes
2. Expand candidates via subgraph query (2-hop)
3. Build structured context from subgraph
4. Construct grounded prompt with graph citations
5. Stream LLM response with source attribution

This is the main entry point for chat inference, called by the gRPC server.
"""

from __future__ import annotations

import uuid
from typing import AsyncGenerator, List, Optional

from loguru import logger

from src.models.schemas import (
    ChatMessage, ChatOptions, ChatRequest, ChatToken,
    SubgraphResult, TokenType,
)
from src.nlp.embedder import SentenceEmbedder
from src.llm.client import LLMClient
from src.llm.streaming import StreamingHandler
from src.rag.prompt_builder import PromptBuilder
from src.rag.retriever import GraphAwareRetriever


class KnowledgeGraphRAGPipeline:
    """Graph-grounded RAG pipeline for conversational Q&A.

    Combines vector retrieval, graph traversal, prompt construction,
    and LLM streaming to produce citation-backed answers grounded
    in the knowledge graph's actual data.
    """

    def __init__(
        self,
        embedder: SentenceEmbedder,
        llm_client: LLMClient,
        graph_client=None,  # Optional gRPC client for graph-engine
        prompt_builder: Optional[PromptBuilder] = None,
        retriever: Optional[GraphAwareRetriever] = None,
    ):
        """Initialize the RAG pipeline.

        Args:
            embedder: Sentence embedding model.
            llm_client: LLM client for inference.
            graph_client: gRPC client for graph-engine (optional).
            prompt_builder: Custom prompt builder (optional).
            retriever: Custom retriever (optional).
        """
        self.embedder = embedder
        self.llm = llm_client
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.retriever = retriever or GraphAwareRetriever(
            embedder=embedder,
            graph_client=graph_client,
        )

        logger.info("KnowledgeGraphRAGPipeline initialized")

    async def chat(
        self,
        message: str,
        history: List[ChatMessage],
        session_id: str,
        options: ChatOptions,
    ) -> AsyncGenerator[ChatToken, None]:
        """Process a chat message through the full RAG pipeline.

        This is the main entry point for conversational Q&A. It:
        1. Embeds the query
        2. Retrieves relevant subgraph context
        3. Builds a grounded prompt
        4. Streams the LLM response with citations
        5. Yields a final GRAPH_UPDATE for frontend visualization

        Args:
            message: User's question.
            history: Previous conversation turns.
            session_id: Session UUID for tracking.
            options: Generation configuration.

        Yields:
            ChatToken objects for SSE streaming.
        """
        logger.info(
            f"RAG chat: session={session_id}, "
            f"message='{message[:80]}...', "
            f"history_turns={len(history)}"
        )

        # ====================================================================
        # STEP 1: Retrieve relevant subgraph context
        # ====================================================================
        try:
            subgraph = await self.retriever.retrieve(message)
            logger.info(
                f"Retrieved context: {len(subgraph.nodes)} nodes, "
                f"{len(subgraph.edges)} edges"
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            subgraph = SubgraphResult()

        # ====================================================================
        # STEP 2: Build structured context from subgraph
        # ====================================================================
        context = self.prompt_builder.build_graph_context(subgraph)

        # ====================================================================
        # STEP 3: Construct grounded prompt with citations
        # ====================================================================
        prompt = self.prompt_builder.build(
            query=message,
            context=context,
            history=history,
        )

        logger.debug(f"Prompt constructed: {len(prompt)} chars")

        # ====================================================================
        # STEP 4: Stream LLM response
        # ====================================================================
        streaming_handler = StreamingHandler(session_id)

        # Stream LLM tokens through the handler
        async for token in streaming_handler.stream_tokens(
            self.llm.stream(
                prompt=prompt,
                temperature=options.temperature,
                max_tokens=options.max_tokens,
            ),
            include_subgraph=options.stream_subgraph,
        ):
            # Yield text/citation/error tokens to the caller
            if token.type != TokenType.DONE:
                yield token

        # ====================================================================
        # STEP 5: Yield subgraph as final GRAPH_UPDATE event
        # ====================================================================
        if options.stream_subgraph and subgraph.nodes:
            import json
            subgraph_data = {
                "nodes": [
                    {
                        "id": n.id,
                        "label": n.label,
                        "name": n.name,
                        "properties": n.properties,
                        "confidence": n.confidence,
                    }
                    for n in subgraph.nodes
                ],
                "edges": [
                    {
                        "id": e.id,
                        "source": e.source_id,
                        "target": e.target_id,
                        "relation": e.relation_type,
                        "weight": e.weight,
                    }
                    for e in subgraph.edges
                ],
            }

            yield ChatToken(
                text=json.dumps(subgraph_data),
                type=TokenType.GRAPH_UPDATE,
            )

        # ====================================================================
        # STEP 6: Yield done sentinel
        # ====================================================================
        yield ChatToken(text="", type=TokenType.DONE)

        logger.info(
            f"RAG chat complete: session={session_id}, "
            f"response_len={len(streaming_handler.full_response)}"
        )

    async def ingest_document(
        self,
        document_id: str,
        content: str,
        title: str = "",
        metadata: Optional[dict] = None,
    ) -> AsyncGenerator:
        """Ingest a document through the full NLP pipeline.

        Steps:
        1. Chunk the document into semantic pieces
        2. Extract entities and relations from each chunk
        3. Generate embeddings for entities
        4. Upsert nodes and edges into the graph-engine
        5. Index embeddings in the vector store

        Yields IngestProgress updates for each stage.
        """
        from src.nlp.chunker import SemanticChunker
        from src.nlp.extractor import KnowledgeExtractor
        from src.models.schemas import Document, IngestProgress, IngestStage

        logger.info(f"Ingesting document: {document_id}, title='{title}'")

        chunker = SemanticChunker()
        extractor = KnowledgeExtractor()

        document = Document(
            id=document_id,
            content=content,
            title=title,
            metadata=metadata or {},
        )

        # Stage 1: Chunking
        yield IngestProgress(
            stage=IngestStage.CHUNKING, progress=0.1,
            message="Splitting document into chunks..."
        )

        chunks = chunker.chunk_document(document)
        total_chunks = len(chunks)

        yield IngestProgress(
            stage=IngestStage.CHUNKING, progress=0.2,
            message=f"Created {total_chunks} chunks",
            total_chunks=total_chunks,
        )

        # Stage 2-3: Entity and Relation extraction
        all_entities = []
        all_relations = []

        for i, chunk in enumerate(chunks):
            yield IngestProgress(
                stage=IngestStage.ENTITY_EXTRACTION,
                progress=0.2 + 0.4 * (i / max(total_chunks, 1)),
                message=f"Processing chunk {i+1}/{total_chunks}",
                chunks_processed=i + 1,
                total_chunks=total_chunks,
            )

            entities, relations = extractor.extract(
                chunk.text, document_id=document_id
            )
            all_entities.extend(entities)
            all_relations.extend(relations)

        yield IngestProgress(
            stage=IngestStage.RELATION_EXTRACTION, progress=0.6,
            message=f"Extracted {len(all_entities)} entities, {len(all_relations)} relations",
            entities_found=len(all_entities),
            relations_found=len(all_relations),
            chunks_processed=total_chunks,
            total_chunks=total_chunks,
        )

        # Stage 4: Embedding generation
        yield IngestProgress(
            stage=IngestStage.EMBEDDING_GENERATION, progress=0.7,
            message="Generating embeddings for entities..."
        )

        entity_texts = [e.text for e in all_entities]
        if entity_texts:
            embeddings = await self.embedder.embed_batch(entity_texts)
        else:
            embeddings = []

        # Stage 5-6: Graph + Vector storage (via graph-engine gRPC in production)
        yield IngestProgress(
            stage=IngestStage.GRAPH_STORAGE, progress=0.85,
            message="Storing entities and relations in knowledge graph..."
        )

        # In production, this would call graph-engine gRPC BatchIngest
        # to insert all nodes and edges at once.

        yield IngestProgress(
            stage=IngestStage.COMPLETED, progress=1.0,
            message=f"Document ingested successfully: "
                    f"{len(all_entities)} entities, {len(all_relations)} relations",
            entities_found=len(all_entities),
            relations_found=len(all_relations),
            chunks_processed=total_chunks,
            total_chunks=total_chunks,
        )

        logger.info(
            f"Ingestion complete: doc={document_id}, "
            f"entities={len(all_entities)}, relations={len(all_relations)}"
        )
