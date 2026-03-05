// Package gateway — graph.go handles graph query and visualization endpoints.
// Proxies graph queries to the Rust graph-engine gRPC service and
// returns serialized subgraphs for the frontend graph visualizer.
package gateway

import (
	"encoding/json"
	"net/http"

	"github.com/go-chi/chi/v5"
	"go.uber.org/zap"

	"github.com/kgchat/api-gateway/internal/clients"
)

// GraphQueryRequest is the JSON body for graph queries.
type GraphQueryRequest struct {
	SeedNodeIDs    []string `json:"seed_node_ids"`
	MaxHops        int      `json:"max_hops"`
	RelationFilter []string `json:"relation_filter,omitempty"`
	MaxNodes       int      `json:"max_nodes,omitempty"`
}

// GraphSearchRequest is the JSON body for vector/hybrid search.
type GraphSearchRequest struct {
	Query       string   `json:"query"`
	K           int      `json:"k,omitempty"`
	LabelFilter []string `json:"label_filter,omitempty"`
	UseHybrid   bool     `json:"use_hybrid,omitempty"`
}

// GraphHandler handles graph query and visualization endpoints.
type GraphHandler struct {
	graphClient *clients.GraphClient
	logger      *zap.Logger
}

// NewGraphHandler creates a new graph handler.
func NewGraphHandler(graphClient *clients.GraphClient, logger *zap.Logger) *GraphHandler {
	return &GraphHandler{
		graphClient: graphClient,
		logger:      logger,
	}
}

// HandleQuery processes a subgraph query request.
// Proxies to graph-engine.QuerySubgraph gRPC.
func (h *GraphHandler) HandleQuery(w http.ResponseWriter, r *http.Request) {
	var req GraphQueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.logger.Error("Failed to decode graph query", zap.Error(err))
		http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	if len(req.SeedNodeIDs) == 0 {
		http.Error(w, `{"error":"seed_node_ids is required"}`, http.StatusBadRequest)
		return
	}

	if req.MaxHops <= 0 {
		req.MaxHops = 2 // Default to 2-hop expansion
	}
	if req.MaxHops > 3 {
		req.MaxHops = 3 // Cap at 3 hops for performance
	}
	if req.MaxNodes <= 0 {
		req.MaxNodes = 50
	}

	h.logger.Info("Graph query request",
		zap.Int("seed_count", len(req.SeedNodeIDs)),
		zap.Int("max_hops", req.MaxHops),
		zap.Int("max_nodes", req.MaxNodes),
	)

	// In production, this would call graph-engine.QuerySubgraph gRPC:
	//   result, err := h.graphClient.QuerySubgraph(ctx, &proto.SubgraphQuery{...})
	//
	// Simulated response for demo:
	response := map[string]interface{}{
		"nodes": []map[string]interface{}{
			{"id": "n1", "label": "CVE", "name": "CVE-2021-44228", "properties": map[string]string{"severity": "CRITICAL"}},
			{"id": "n2", "label": "SOFTWARE", "name": "Apache Log4j 2", "properties": map[string]string{"version": "2.14.1"}},
			{"id": "n3", "label": "THREAT_ACTOR", "name": "APT41", "properties": map[string]string{"origin": "China"}},
			{"id": "n4", "label": "ATTACK_PATTERN", "name": "Remote Code Execution", "properties": map[string]string{}},
			{"id": "n5", "label": "MITIGATION", "name": "Upgrade to Log4j 2.17.0", "properties": map[string]string{}},
		},
		"edges": []map[string]interface{}{
			{"source": "n1", "target": "n2", "relation": "AFFECTS", "weight": 1.0},
			{"source": "n3", "target": "n1", "relation": "EXPLOITS", "weight": 0.95},
			{"source": "n1", "target": "n4", "relation": "ENABLES", "weight": 0.9},
			{"source": "n5", "target": "n1", "relation": "MITIGATES", "weight": 1.0},
		},
		"total_nodes_visited": 5,
		"traversal_time_ms":   2.3,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// HandleGetNode retrieves a single node by its ID.
func (h *GraphHandler) HandleGetNode(w http.ResponseWriter, r *http.Request) {
	nodeID := chi.URLParam(r, "nodeID")
	if nodeID == "" {
		http.Error(w, `{"error":"nodeID parameter is required"}`, http.StatusBadRequest)
		return
	}

	h.logger.Info("Get node request", zap.String("node_id", nodeID))

	// In production: graphClient.GetNode(ctx, &proto.GetNodeRequest{NodeId: nodeID})
	// Simulated response:
	response := map[string]interface{}{
		"node": map[string]interface{}{
			"id":         nodeID,
			"label":      "CVE",
			"name":       "CVE-2021-44228",
			"properties": map[string]string{"severity": "CRITICAL", "cvss": "10.0", "description": "Apache Log4j2 RCE"},
			"confidence": 0.98,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// HandleStats returns graph statistics.
func (h *GraphHandler) HandleStats(w http.ResponseWriter, r *http.Request) {
	h.logger.Info("Graph stats request")

	// In production: graphClient.GetStats(ctx, &proto.StatsRequest{})
	response := map[string]interface{}{
		"total_nodes":       156,
		"total_edges":       312,
		"vector_index_size": 148,
		"nodes_by_label": map[string]int{
			"CVE":            45,
			"SOFTWARE":       38,
			"THREAT_ACTOR":   22,
			"ORGANIZATION":   18,
			"MITIGATION":     15,
			"ATTACK_PATTERN": 12,
			"VULNERABILITY":  6,
		},
		"edges_by_type": map[string]int{
			"AFFECTS":       78,
			"EXPLOITS":      52,
			"MITIGATES":     41,
			"USES":          38,
			"TARGETS":       35,
			"ATTRIBUTED_TO": 28,
			"RELATED_TO":    25,
			"DISCOVERED_BY": 15,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// HandleSearch performs a vector or hybrid search query.
func (h *GraphHandler) HandleSearch(w http.ResponseWriter, r *http.Request) {
	var req GraphSearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.logger.Error("Failed to decode search request", zap.Error(err))
		http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	if req.Query == "" {
		http.Error(w, `{"error":"query is required"}`, http.StatusBadRequest)
		return
	}

	if req.K <= 0 {
		req.K = 10
	}

	h.logger.Info("Graph search request",
		zap.String("query", req.Query),
		zap.Int("k", req.K),
		zap.Bool("hybrid", req.UseHybrid),
	)

	// In production: embed query via ai-pipeline, then vector/hybrid search via graph-engine
	response := map[string]interface{}{
		"results": []map[string]interface{}{
			{"node_id": "n1", "name": "CVE-2021-44228", "label": "CVE", "score": 0.95},
			{"node_id": "n2", "name": "Apache Log4j 2", "label": "SOFTWARE", "score": 0.88},
		},
		"search_time_ms": 4.2,
		"total_results":  2,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}
