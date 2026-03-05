// Package middleware — CORS middleware for cross-origin requests.
// Configured with allowed origins from the application config.
package middleware

import (
	"net/http"
	"strings"
)

// CORSMiddleware handles Cross-Origin Resource Sharing headers.
type CORSMiddleware struct {
	allowedOrigins map[string]bool
	allowAll       bool
}

// NewCORSMiddleware creates a CORS middleware with the given allowed origins.
// If origins contains "*", all origins are allowed (development only).
func NewCORSMiddleware(origins []string) *CORSMiddleware {
	m := &CORSMiddleware{
		allowedOrigins: make(map[string]bool),
	}

	for _, origin := range origins {
		if origin == "*" {
			m.allowAll = true
			break
		}
		m.allowedOrigins[strings.TrimRight(origin, "/")] = true
	}

	return m
}

// Handler returns the CORS middleware handler.
func (m *CORSMiddleware) Handler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")

		// Check if the origin is allowed
		if m.allowAll || m.allowedOrigins[origin] {
			w.Header().Set("Access-Control-Allow-Origin", origin)
		} else if origin != "" {
			// For non-matching origins, we still need to handle the request
			// but don't set CORS headers — the browser will block it
		}

		// Always set these headers for CORS compatibility
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS, PATCH")
		w.Header().Set("Access-Control-Allow-Headers", "Accept, Authorization, Content-Type, X-Request-ID, X-Session-ID")
		w.Header().Set("Access-Control-Allow-Credentials", "true")
		w.Header().Set("Access-Control-Max-Age", "86400") // 24 hours preflight cache
		w.Header().Set("Access-Control-Expose-Headers", "X-Request-ID")

		// Handle preflight OPTIONS requests immediately
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}
