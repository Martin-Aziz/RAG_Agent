// Package clients provides gRPC client factories for downstream services.
// graph_client.go creates a typed client for the Rust graph-engine service.
package clients

import (
	"context"
	"fmt"
	"time"

	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
)

// GraphClient wraps a gRPC connection to the graph-engine service.
// It provides typed methods that map to GraphService RPCs.
type GraphClient struct {
	conn   *grpc.ClientConn
	logger *zap.Logger
}

// NewGraphClient creates a new gRPC client connection to the graph-engine.
// Uses insecure transport (TLS should be added for production deployment).
func NewGraphClient(addr string, dialTimeout time.Duration, logger *zap.Logger) (*GraphClient, error) {
	ctx, cancel := context.WithTimeout(context.Background(), dialTimeout)
	defer cancel()

	// Configure keepalive to detect dead connections
	kaParams := keepalive.ClientParameters{
		Time:                10 * time.Second,
		Timeout:             3 * time.Second,
		PermitWithoutStream: true,
	}

	conn, err := grpc.DialContext(ctx, addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(kaParams),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(50*1024*1024), // 50MB max response
		),
	)
	if err != nil {
		return nil, fmt.Errorf("connecting to graph-engine at %s: %w", addr, err)
	}

	logger.Info("Connected to graph-engine", zap.String("addr", addr))

	return &GraphClient{
		conn:   conn,
		logger: logger,
	}, nil
}

// Conn returns the underlying gRPC connection for use with generated stubs.
func (c *GraphClient) Conn() *grpc.ClientConn {
	return c.conn
}

// Close closes the gRPC connection.
func (c *GraphClient) Close() error {
	c.logger.Info("Closing graph-engine connection")
	return c.conn.Close()
}
