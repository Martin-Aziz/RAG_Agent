// Package middleware provides HTTP middleware for the API gateway.
// auth.go implements JWT-based authentication and authorization.
package middleware

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/golang-jwt/jwt/v5"
	"go.uber.org/zap"
)

// contextKey is a typed key for storing values in request context.
// Using a custom type prevents collisions with other packages.
type contextKey string

const (
	// UserIDKey is the context key for the authenticated user's ID.
	UserIDKey contextKey = "user_id"
	// SessionIDKey is the context key for the user's session ID.
	SessionIDKey contextKey = "session_id"
)

// Claims represents the JWT token payload with standard and custom fields.
type Claims struct {
	UserID    string `json:"user_id"`
	SessionID string `json:"session_id"`
	jwt.RegisteredClaims
}

// AuthMiddleware validates JWT tokens from the Authorization header.
// If auth is disabled (dev mode), it injects default claims.
type AuthMiddleware struct {
	secret  []byte
	enabled bool
	logger  *zap.Logger
}

// NewAuthMiddleware creates a new JWT authentication middleware.
func NewAuthMiddleware(secret string, enabled bool, logger *zap.Logger) *AuthMiddleware {
	return &AuthMiddleware{
		secret:  []byte(secret),
		enabled: enabled,
		logger:  logger,
	}
}

// Handler returns the middleware handler function.
func (m *AuthMiddleware) Handler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// If auth is disabled, inject default dev claims and proceed
		if !m.enabled {
			ctx := context.WithValue(r.Context(), UserIDKey, "dev-user")
			ctx = context.WithValue(ctx, SessionIDKey, "dev-session")
			next.ServeHTTP(w, r.WithContext(ctx))
			return
		}

		// Extract token from Authorization: Bearer <token> header
		authHeader := r.Header.Get("Authorization")
		if authHeader == "" {
			http.Error(w, `{"error":"missing authorization header"}`, http.StatusUnauthorized)
			return
		}

		parts := strings.SplitN(authHeader, " ", 2)
		if len(parts) != 2 || strings.ToLower(parts[0]) != "bearer" {
			http.Error(w, `{"error":"invalid authorization format, expected Bearer <token>"}`, http.StatusUnauthorized)
			return
		}

		tokenString := parts[1]

		// Parse and validate the JWT token
		claims := &Claims{}
		token, err := jwt.ParseWithClaims(tokenString, claims, func(token *jwt.Token) (interface{}, error) {
			// Verify the signing method is HMAC
			if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
				return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
			}
			return m.secret, nil
		})

		if err != nil {
			m.logger.Warn("JWT validation failed",
				zap.Error(err),
				zap.String("remote_addr", r.RemoteAddr),
			)
			http.Error(w, `{"error":"invalid or expired token"}`, http.StatusUnauthorized)
			return
		}

		if !token.Valid {
			http.Error(w, `{"error":"invalid token"}`, http.StatusUnauthorized)
			return
		}

		// Inject claims into request context for downstream handlers
		ctx := context.WithValue(r.Context(), UserIDKey, claims.UserID)
		ctx = context.WithValue(ctx, SessionIDKey, claims.SessionID)

		m.logger.Debug("Request authenticated",
			zap.String("user_id", claims.UserID),
			zap.String("session_id", claims.SessionID),
		)

		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// GetUserID extracts the user ID from the request context.
func GetUserID(ctx context.Context) string {
	if val, ok := ctx.Value(UserIDKey).(string); ok {
		return val
	}
	return ""
}

// GetSessionID extracts the session ID from the request context.
func GetSessionID(ctx context.Context) string {
	if val, ok := ctx.Value(SessionIDKey).(string); ok {
		return val
	}
	return ""
}
