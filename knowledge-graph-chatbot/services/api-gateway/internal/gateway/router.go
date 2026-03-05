// Package gateway provides HTTP handlers for the API gateway.
// router.go sets up the Chi router with all middleware and route registrations.
package gateway

import (
	"net/http"

	"github.com/go-chi/chi/v5"
	chimiddleware "github.com/go-chi/chi/v5/middleware"
	"go.uber.org/zap"

	"github.com/kgchat/api-gateway/internal/clients"
	"github.com/kgchat/api-gateway/internal/middleware"
	"github.com/kgchat/api-gateway/internal/ws"
)

// RouterDeps holds all dependencies needed to configure the HTTP router.
type RouterDeps struct {
	GraphClient *clients.GraphClient
	AIClient    *clients.AIClient
	WSHub       *ws.Hub
	Auth        *middleware.AuthMiddleware
	RateLimiter *middleware.RateLimiter
	CORS        *middleware.CORSMiddleware
	Logger      *zap.Logger
}

// NewRouter creates and configures the Chi router with all routes and middleware.
func NewRouter(deps RouterDeps) http.Handler {
	r := chi.NewRouter()

	// ========================================================================
	// Global middleware stack (applied to all routes)
	// ========================================================================
	r.Use(chimiddleware.RequestID)    // Inject X-Request-ID header
	r.Use(chimiddleware.RealIP)       // Extract real IP from proxy headers
	r.Use(chimiddleware.Recoverer)    // Recover from panics with 500 response
	r.Use(deps.CORS.Handler)          // CORS headers for cross-origin requests
	r.Use(requestLogger(deps.Logger)) // Structured request/response logging

	// ========================================================================
	// Health check — unauthenticated, unrated
	// ========================================================================
	r.Get("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"ok","service":"api-gateway"}`))
	})

	// ========================================================================
	// API routes — authenticated
	// ========================================================================
	r.Group(func(r chi.Router) {
		r.Use(deps.Auth.Handler) // JWT authentication

		// Chat endpoints — rate limited at 10/min per user
		r.Group(func(r chi.Router) {
			r.Use(deps.RateLimiter.ChatLimitHandler)

			chatHandler := NewChatHandler(deps.AIClient, deps.WSHub, deps.Logger)
			r.Post("/api/v1/chat", chatHandler.HandleChat)
			r.Get("/api/v1/chat/sessions", chatHandler.ListSessions)
		})

		// Document ingestion endpoints
		r.Group(func(r chi.Router) {
			ingestHandler := NewIngestHandler(deps.AIClient, deps.Logger)
			r.Post("/api/v1/ingest", ingestHandler.HandleIngest)
			r.Post("/api/v1/ingest/batch", ingestHandler.HandleBatchIngest)
		})

		// Graph query endpoints — rate limited at 100/min per user
		r.Group(func(r chi.Router) {
			r.Use(deps.RateLimiter.GraphLimitHandler)

			graphHandler := NewGraphHandler(deps.GraphClient, deps.Logger)
			r.Post("/api/v1/graph/query", graphHandler.HandleQuery)
			r.Get("/api/v1/graph/node/{nodeID}", graphHandler.HandleGetNode)
			r.Get("/api/v1/graph/stats", graphHandler.HandleStats)
			r.Post("/api/v1/graph/search", graphHandler.HandleSearch)
		})

		// WebSocket endpoint for real-time graph updates
		r.Get("/ws", deps.WSHub.ServeWS)
	})

	return r
}

// requestLogger returns a middleware that logs each HTTP request with zap.
func requestLogger(logger *zap.Logger) func(next http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ww := chimiddleware.NewWrapResponseWriter(w, r.ProtoMajor)

			defer func() {
				logger.Info("HTTP request",
					zap.String("method", r.Method),
					zap.String("path", r.URL.Path),
					zap.Int("status", ww.Status()),
					zap.Int("bytes", ww.BytesWritten()),
					zap.String("remote_addr", r.RemoteAddr),
					zap.String("request_id", chimiddleware.GetReqID(r.Context())),
				)
			}()

			next.ServeHTTP(ww, r)
		})
	}
}
