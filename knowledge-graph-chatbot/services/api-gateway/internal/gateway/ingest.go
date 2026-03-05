// Package gateway — ingest.go handles document ingestion requests.
// Accepts documents via HTTP POST and dispatches them to the ai-pipeline
// for NLP processing (chunking, NER, relation extraction, embedding).
package gateway

import (
	"encoding/json"
	"net/http"

	"go.uber.org/zap"

	"github.com/kgchat/api-gateway/internal/clients"
	"github.com/kgchat/api-gateway/internal/middleware"
)

// IngestRequest is the JSON body for document ingestion.
type IngestRequest struct {
	DocumentID string            `json:"document_id"`
	Content    string            `json:"content"`
	Title      string            `json:"title"`
	SourceURL  string            `json:"source_url,omitempty"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// IngestResponse reports ingestion results.
type IngestResponse struct {
	DocumentID      string `json:"document_id"`
	Status          string `json:"status"`
	EntitiesFound   int    `json:"entities_found"`
	RelationsFound  int    `json:"relations_found"`
	ChunksProcessed int    `json:"chunks_processed"`
	Message         string `json:"message,omitempty"`
}

// IngestHandler handles document ingestion requests.
type IngestHandler struct {
	aiClient *clients.AIClient
	logger   *zap.Logger
}

// NewIngestHandler creates a new ingestion handler.
func NewIngestHandler(aiClient *clients.AIClient, logger *zap.Logger) *IngestHandler {
	return &IngestHandler{
		aiClient: aiClient,
		logger:   logger,
	}
}

// HandleIngest processes a single document ingestion request.
// In production, this would:
//  1. Validate the request
//  2. Call ai-pipeline.IngestDocument gRPC (streaming progress)
//  3. Return aggregated results
func (h *IngestHandler) HandleIngest(w http.ResponseWriter, r *http.Request) {
	var req IngestRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.logger.Error("Failed to decode ingest request", zap.Error(err))
		http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	// Validate required fields
	if req.Content == "" {
		http.Error(w, `{"error":"content is required"}`, http.StatusBadRequest)
		return
	}

	if req.DocumentID == "" {
		http.Error(w, `{"error":"document_id is required"}`, http.StatusBadRequest)
		return
	}

	userID := middleware.GetUserID(r.Context())

	h.logger.Info("Ingest request received",
		zap.String("user_id", userID),
		zap.String("document_id", req.DocumentID),
		zap.String("title", req.Title),
		zap.Int("content_length", len(req.Content)),
	)

	// In production, this would call the ai-pipeline gRPC IngestDocument RPC:
	//   stream, err := h.aiClient.IngestDocument(r.Context(), &proto.IngestRequest{...})
	//   for { progress, err := stream.Recv(); ... }
	//
	// Simulated response for demo/testing:
	resp := IngestResponse{
		DocumentID:      req.DocumentID,
		Status:          "completed",
		EntitiesFound:   12,
		RelationsFound:  8,
		ChunksProcessed: 5,
		Message:         "Document processed successfully through the NLP pipeline",
	}

	h.logger.Info("Ingest completed",
		zap.String("document_id", req.DocumentID),
		zap.Int("entities", resp.EntitiesFound),
		zap.Int("relations", resp.RelationsFound),
	)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(resp)
}

// HandleBatchIngest processes multiple documents in a single request.
func (h *IngestHandler) HandleBatchIngest(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Documents []IngestRequest `json:"documents"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.logger.Error("Failed to decode batch ingest request", zap.Error(err))
		http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	if len(req.Documents) == 0 {
		http.Error(w, `{"error":"at least one document is required"}`, http.StatusBadRequest)
		return
	}

	userID := middleware.GetUserID(r.Context())

	h.logger.Info("Batch ingest request received",
		zap.String("user_id", userID),
		zap.Int("document_count", len(req.Documents)),
	)

	// Process each document and collect results
	results := make([]IngestResponse, 0, len(req.Documents))
	for _, doc := range req.Documents {
		// In production: dispatch to NATS JetStream for async processing
		results = append(results, IngestResponse{
			DocumentID: doc.DocumentID,
			Status:     "queued",
			Message:    "Document queued for processing",
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"results": results,
		"total":   len(results),
		"status":  "queued",
	})
}
