// Package main — API Gateway service entrypoint.
// Initializes gRPC clients, middleware, WebSocket hub, and HTTP server.
package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"go.uber.org/zap"

	"github.com/kgchat/api-gateway/internal/clients"
	"github.com/kgchat/api-gateway/internal/config"
	"github.com/kgchat/api-gateway/internal/gateway"
	"github.com/kgchat/api-gateway/internal/middleware"
	"github.com/kgchat/api-gateway/internal/ws"
)

func main() {
	// ========================================================================
	// 1. Initialize structured logger
	// ========================================================================
	logger, err := zap.NewProduction()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create logger: %v\n", err)
		os.Exit(1)
	}
	defer logger.Sync()

	logger.Info("Starting API gateway service")

	// ========================================================================
	// 2. Load configuration
	// ========================================================================
	cfg, err := config.Load()
	if err != nil {
		logger.Fatal("Failed to load configuration", zap.Error(err))
	}

	logger.Info("Configuration loaded",
		zap.Int("port", cfg.Server.Port),
		zap.String("graph_engine", cfg.GRPC.GraphEngineAddr),
		zap.String("ai_pipeline", cfg.GRPC.AIPipelineAddr),
		zap.Bool("auth_enabled", cfg.Auth.Enabled),
	)

	// ========================================================================
	// 3. Initialize gRPC clients
	// ========================================================================
	graphClient, err := clients.NewGraphClient(
		cfg.GRPC.GraphEngineAddr,
		cfg.GRPC.DialTimeout,
		logger,
	)
	if err != nil {
		logger.Fatal("Failed to connect to graph-engine", zap.Error(err))
	}
	defer graphClient.Close()

	aiClient, err := clients.NewAIClient(
		cfg.GRPC.AIPipelineAddr,
		cfg.GRPC.DialTimeout,
		logger,
	)
	if err != nil {
		logger.Fatal("Failed to connect to ai-pipeline", zap.Error(err))
	}
	defer aiClient.Close()

	// ========================================================================
	// 4. Initialize middleware
	// ========================================================================
	authMiddleware := middleware.NewAuthMiddleware(
		cfg.Auth.JWTSecret,
		cfg.Auth.Enabled,
		logger,
	)

	rateLimiter := middleware.NewRateLimiter(
		middleware.DefaultRateLimitConfig(),
		logger,
	)

	corsMiddleware := middleware.NewCORSMiddleware(cfg.Server.CORSOrigins)

	// ========================================================================
	// 5. Initialize WebSocket hub
	// ========================================================================
	wsHub := ws.NewHub(logger)
	go wsHub.Run()

	// ========================================================================
	// 6. Create HTTP router with all routes and middleware
	// ========================================================================
	router := gateway.NewRouter(gateway.RouterDeps{
		GraphClient: graphClient,
		AIClient:    aiClient,
		WSHub:       wsHub,
		Auth:        authMiddleware,
		RateLimiter: rateLimiter,
		CORS:        corsMiddleware,
		Logger:      logger,
	})

	// ========================================================================
	// 7. Start HTTP server with graceful shutdown
	// ========================================================================
	addr := fmt.Sprintf(":%d", cfg.Server.Port)
	srv := &http.Server{
		Addr:         addr,
		Handler:      router,
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
		IdleTimeout:  120 * time.Second,
	}

	// Start server in a goroutine
	go func() {
		logger.Info("HTTP server listening", zap.String("addr", addr))
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("HTTP server error", zap.Error(err))
		}
	}()

	// Wait for interrupt signal for graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("Shutdown signal received, draining connections...")

	// Give outstanding requests time to complete
	ctx, cancel := context.WithTimeout(context.Background(), cfg.Server.ShutdownTimeout)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		logger.Fatal("Server forced to shutdown", zap.Error(err))
	}

	logger.Info("API gateway shut down gracefully")
}
