// Package middleware — rate limiter using token bucket algorithm.
// Enforces per-user request limits to prevent abuse:
// - 10 chat requests/minute per user
// - 100 graph queries/minute per user
package middleware

import (
	"net/http"
	"sync"

	"go.uber.org/zap"
	"golang.org/x/time/rate"
)

// RateLimitConfig defines rate limiting parameters for different endpoint types.
type RateLimitConfig struct {
	// ChatRate is the max chat requests per second (10/min = 0.167/sec)
	ChatRate rate.Limit
	// ChatBurst is the max burst size for chat requests
	ChatBurst int
	// GraphRate is the max graph queries per second (100/min = 1.667/sec)
	GraphRate rate.Limit
	// GraphBurst is the max burst size for graph queries
	GraphBurst int
}

// DefaultRateLimitConfig returns production rate limit values.
func DefaultRateLimitConfig() RateLimitConfig {
	return RateLimitConfig{
		ChatRate:   rate.Limit(10.0 / 60.0),  // 10 per minute
		ChatBurst:  3,                        // Allow short bursts of 3
		GraphRate:  rate.Limit(100.0 / 60.0), // 100 per minute
		GraphBurst: 10,                       // Allow short bursts of 10
	}
}

// RateLimiter manages per-user rate limiters for different endpoint types.
type RateLimiter struct {
	// chatLimiters stores per-user token bucket limiters for chat endpoints.
	// sync.Map is used for safe concurrent access without a global mutex.
	chatLimiters sync.Map
	// graphLimiters stores per-user limiters for graph query endpoints.
	graphLimiters sync.Map
	config        RateLimitConfig
	logger        *zap.Logger
}

// NewRateLimiter creates a new rate limiter with the given configuration.
func NewRateLimiter(config RateLimitConfig, logger *zap.Logger) *RateLimiter {
	return &RateLimiter{
		config: config,
		logger: logger,
	}
}

// getLimiter retrieves or creates a rate limiter for the given user and store.
func getLimiter(store *sync.Map, userID string, r rate.Limit, burst int) *rate.Limiter {
	if limiter, ok := store.Load(userID); ok {
		return limiter.(*rate.Limiter)
	}
	// Create a new limiter for this user. race-safe: LoadOrStore handles concurrent creation.
	limiter := rate.NewLimiter(r, burst)
	actual, _ := store.LoadOrStore(userID, limiter)
	return actual.(*rate.Limiter)
}

// ChatLimitHandler returns middleware that rate-limits chat requests per user.
func (rl *RateLimiter) ChatLimitHandler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		userID := GetUserID(r.Context())
		if userID == "" {
			userID = r.RemoteAddr // Fallback to IP if no auth
		}

		limiter := getLimiter(&rl.chatLimiters, userID, rl.config.ChatRate, rl.config.ChatBurst)

		if !limiter.Allow() {
			rl.logger.Warn("Chat rate limit exceeded",
				zap.String("user_id", userID),
				zap.String("remote_addr", r.RemoteAddr),
			)

			w.Header().Set("Retry-After", "60")
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusTooManyRequests)
			w.Write([]byte(`{"error":"rate limit exceeded","retry_after_seconds":60}`))
			return
		}

		next.ServeHTTP(w, r)
	})
}

// GraphLimitHandler returns middleware that rate-limits graph query requests per user.
func (rl *RateLimiter) GraphLimitHandler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		userID := GetUserID(r.Context())
		if userID == "" {
			userID = r.RemoteAddr
		}

		limiter := getLimiter(&rl.graphLimiters, userID, rl.config.GraphRate, rl.config.GraphBurst)

		if !limiter.Allow() {
			rl.logger.Warn("Graph query rate limit exceeded",
				zap.String("user_id", userID),
				zap.String("remote_addr", r.RemoteAddr),
			)

			w.Header().Set("Retry-After", "30")
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusTooManyRequests)
			w.Write([]byte(`{"error":"rate limit exceeded","retry_after_seconds":30}`))
			return
		}

		next.ServeHTTP(w, r)
	})
}
