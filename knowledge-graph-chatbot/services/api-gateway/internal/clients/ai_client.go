// Package clients — ai_client.go creates a typed client for the Python ai-pipeline.
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

// AIClient wraps a gRPC connection to the ai-pipeline service.
type AIClient struct {
	conn   *grpc.ClientConn
	logger *zap.Logger
}

// NewAIClient creates a new gRPC client connection to the ai-pipeline.
func NewAIClient(addr string, dialTimeout time.Duration, logger *zap.Logger) (*AIClient, error) {
	ctx, cancel := context.WithTimeout(context.Background(), dialTimeout)
	defer cancel()

	kaParams := keepalive.ClientParameters{
		Time:                10 * time.Second,
		Timeout:             3 * time.Second,
		PermitWithoutStream: true,
	}

	conn, err := grpc.DialContext(ctx, addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(kaParams),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(50*1024*1024), // 50MB for large embeddings
			grpc.MaxCallSendMsgSize(50*1024*1024), // 50MB for document ingestion
		),
	)
	if err != nil {
		return nil, fmt.Errorf("connecting to ai-pipeline at %s: %w", addr, err)
	}

	logger.Info("Connected to ai-pipeline", zap.String("addr", addr))

	return &AIClient{
		conn:   conn,
		logger: logger,
	}, nil
}

// Conn returns the underlying gRPC connection for use with generated stubs.
func (c *AIClient) Conn() *grpc.ClientConn {
	return c.conn
}

// Close closes the gRPC connection.
func (c *AIClient) Close() error {
	c.logger.Info("Closing ai-pipeline connection")
	return c.conn.Close()
}
