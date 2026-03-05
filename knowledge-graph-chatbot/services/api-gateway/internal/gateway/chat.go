// Package gateway — chat.go implements the SSE streaming chat endpoint.
// Proxies chat requests to the ai-pipeline gRPC service and streams
// tokens back as Server-Sent Events for real-time UI updates.
package gateway

import (
	"encoding/json"
	"fmt"
	"net/http"

	"go.uber.org/zap"

	"github.com/kgchat/api-gateway/internal/clients"
	"github.com/kgchat/api-gateway/internal/middleware"
	"github.com/kgchat/api-gateway/internal/ws"
)

// ChatRequest is the JSON body for the /api/v1/chat endpoint.
type ChatRequest struct {
	Message   string        `json:"message"`
	SessionID string        `json:"session_id"`
	History   []ChatMessage `json:"history,omitempty"`
	Options   *ChatOptions  `json:"options,omitempty"`
}

// ChatMessage represents a single conversation turn.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatOptions configures LLM generation behavior.
type ChatOptions struct {
	Temperature          float64 `json:"temperature,omitempty"`
	MaxTokens            int     `json:"max_tokens,omitempty"`
	IncludeGraphCitation bool    `json:"include_graph_citation,omitempty"`
	StreamSubgraph       bool    `json:"stream_subgraph,omitempty"`
}

// ChatHandler handles SSE streaming chat requests.
type ChatHandler struct {
	aiClient *clients.AIClient
	wsHub    *ws.Hub
	logger   *zap.Logger
}

// NewChatHandler creates a new chat handler with AI client and WebSocket hub.
func NewChatHandler(aiClient *clients.AIClient, wsHub *ws.Hub, logger *zap.Logger) *ChatHandler {
	return &ChatHandler{
		aiClient: aiClient,
		wsHub:    wsHub,
		logger:   logger,
	}
}

// HandleChat processes a chat request and streams the response as SSE events.
//
// SSE Event Types:
//   - "data: {token}\n\n"          — Regular text token
//   - "event: graph\ndata: {...}\n\n" — Subgraph update for visualization
//   - "event: done\ndata: [DONE]\n\n" — End-of-stream sentinel
//   - "event: error\ndata: {...}\n\n" — Error message
func (h *ChatHandler) HandleChat(w http.ResponseWriter, r *http.Request) {
	// 1. Parse the JSON request body
	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.logger.Error("Failed to decode chat request", zap.Error(err))
		http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	if req.Message == "" {
		http.Error(w, `{"error":"message is required"}`, http.StatusBadRequest)
		return
	}

	// Use session ID from request or from auth context
	if req.SessionID == "" {
		req.SessionID = middleware.GetSessionID(r.Context())
	}

	userID := middleware.GetUserID(r.Context())

	h.logger.Info("Chat request received",
		zap.String("user_id", userID),
		zap.String("session_id", req.SessionID),
		zap.Int("message_length", len(req.Message)),
		zap.Int("history_length", len(req.History)),
	)

	// 2. Set SSE response headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no") // Disable nginx buffering

	flusher, ok := w.(http.Flusher)
	if !ok {
		h.logger.Error("Streaming not supported by response writer")
		http.Error(w, `{"error":"streaming not supported"}`, http.StatusInternalServerError)
		return
	}

	// 3. In a production system, we would call the ai-pipeline gRPC Chat RPC here.
	// For now, we simulate the streaming response pattern.
	//
	// The actual gRPC call would look like:
	//   stream, err := h.aiClient.Chat(r.Context(), &proto.ChatRequest{...})
	//   for { token, err := stream.Recv(); ... }
	//
	// Simulated response for demo/testing:
	response := fmt.Sprintf("I received your question about: \"%s\". "+
		"Based on the knowledge graph, here is what I found:\n\n"+
		"The system is processing your query through the RAG pipeline, "+
		"which involves embedding your question, searching the vector index "+
		"for relevant entities, expanding to a 2-hop subgraph, and generating "+
		"a grounded response with citations.\n\n"+
		"[NODE:demo-node-1] This is a demonstration response. "+
		"In production, this would be streamed from the LLM via the ai-pipeline gRPC service.",
		req.Message)

	// Stream tokens (simulate word-by-word streaming)
	words := splitIntoTokens(response)
	for _, word := range words {
		// Check if client disconnected
		select {
		case <-r.Context().Done():
			h.logger.Info("Client disconnected during streaming",
				zap.String("session_id", req.SessionID),
			)
			return
		default:
		}

		// Write SSE text event
		fmt.Fprintf(w, "data: %s\n\n", word)
		flusher.Flush()
	}

	// 4. Send a mock subgraph update for the frontend graph visualization
	subgraph := map[string]interface{}{
		"nodes": []map[string]interface{}{
			{"id": "demo-node-1", "label": "CVE", "name": "CVE-2021-44228", "confidence": 0.95},
			{"id": "demo-node-2", "label": "SOFTWARE", "name": "Apache Log4j", "confidence": 0.92},
		},
		"edges": []map[string]interface{}{
			{"source": "demo-node-1", "target": "demo-node-2", "relation": "AFFECTS"},
		},
	}
	subgraphJSON, _ := json.Marshal(subgraph)
	fmt.Fprintf(w, "event: graph\ndata: %s\n\n", string(subgraphJSON))
	flusher.Flush()

	// 5. Send done sentinel
	fmt.Fprintf(w, "event: done\ndata: [DONE]\n\n")
	flusher.Flush()

	// 6. Broadcast graph update via WebSocket for real-time visualization
	if h.wsHub != nil {
		h.wsHub.BroadcastGraphUpdate(ws.GraphUpdate{
			Type:      "subgraph",
			SessionID: req.SessionID,
			Payload:   subgraphJSON,
		})
	}

	h.logger.Info("Chat response completed",
		zap.String("session_id", req.SessionID),
		zap.String("user_id", userID),
	)
}

// ListSessions returns the available chat sessions for the authenticated user.
func (h *ChatHandler) ListSessions(w http.ResponseWriter, r *http.Request) {
	userID := middleware.GetUserID(r.Context())

	// In production, this would query PostgreSQL for stored sessions.
	sessions := []map[string]interface{}{
		{
			"id":         "demo-session",
			"title":      "Demo Session",
			"created_at": "2024-01-01T00:00:00Z",
			"user_id":    userID,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"sessions": sessions,
	})
}

// splitIntoTokens splits text into word-level tokens for streaming simulation.
func splitIntoTokens(text string) []string {
	var tokens []string
	current := ""
	for _, ch := range text {
		current += string(ch)
		if ch == ' ' || ch == '\n' {
			tokens = append(tokens, current)
			current = ""
		}
	}
	if current != "" {
		tokens = append(tokens, current)
	}
	return tokens
}
