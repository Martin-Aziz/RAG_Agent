// Package config provides application configuration loading via Viper.
// Reads from environment variables, config files, and sensible defaults.
// All configuration fields are strongly typed for compile-time safety.
package config

import (
	"fmt"
	"strings"
	"time"

	"github.com/spf13/viper"
)

// Config holds all application configuration values.
type Config struct {
	Server   ServerConfig   `mapstructure:"server"`
	GRPC     GRPCConfig     `mapstructure:"grpc"`
	Auth     AuthConfig     `mapstructure:"auth"`
	Redis    RedisConfig    `mapstructure:"redis"`
	Postgres PostgresConfig `mapstructure:"postgres"`
}

// ServerConfig holds HTTP server settings.
type ServerConfig struct {
	Port            int           `mapstructure:"port"`
	ReadTimeout     time.Duration `mapstructure:"read_timeout"`
	WriteTimeout    time.Duration `mapstructure:"write_timeout"`
	ShutdownTimeout time.Duration `mapstructure:"shutdown_timeout"`
	CORSOrigins     []string      `mapstructure:"cors_origins"`
}

// GRPCConfig holds addresses for downstream gRPC services.
type GRPCConfig struct {
	GraphEngineAddr string        `mapstructure:"graph_engine_addr"`
	AIPipelineAddr  string        `mapstructure:"ai_pipeline_addr"`
	DialTimeout     time.Duration `mapstructure:"dial_timeout"`
}

// AuthConfig holds JWT authentication settings.
type AuthConfig struct {
	JWTSecret   string        `mapstructure:"jwt_secret"`
	TokenExpiry time.Duration `mapstructure:"token_expiry"`
	Enabled     bool          `mapstructure:"enabled"`
}

// RedisConfig holds Redis connection settings.
type RedisConfig struct {
	URL      string `mapstructure:"url"`
	Password string `mapstructure:"password"`
	DB       int    `mapstructure:"db"`
}

// PostgresConfig holds PostgreSQL connection settings.
type PostgresConfig struct {
	URL          string `mapstructure:"url"`
	MaxOpenConns int    `mapstructure:"max_open_conns"`
	MaxIdleConns int    `mapstructure:"max_idle_conns"`
}

// Load reads configuration from environment variables and optional config file.
// Environment variables take precedence over config file values.
func Load() (*Config, error) {
	v := viper.New()

	// Set defaults — sensible production values
	v.SetDefault("server.port", 8080)
	v.SetDefault("server.read_timeout", 30*time.Second)
	v.SetDefault("server.write_timeout", 60*time.Second)
	v.SetDefault("server.shutdown_timeout", 10*time.Second)
	v.SetDefault("server.cors_origins", []string{"http://localhost:3000"})

	v.SetDefault("grpc.graph_engine_addr", "graph-engine:50051")
	v.SetDefault("grpc.ai_pipeline_addr", "ai-pipeline:50052")
	v.SetDefault("grpc.dial_timeout", 10*time.Second)

	v.SetDefault("auth.jwt_secret", "change_me_in_production")
	v.SetDefault("auth.token_expiry", 24*time.Hour)
	v.SetDefault("auth.enabled", false) // Disabled by default for development

	v.SetDefault("redis.url", "redis:6379")
	v.SetDefault("redis.password", "")
	v.SetDefault("redis.db", 0)

	v.SetDefault("postgres.url", "postgresql://user:pass@postgres:5432/kgchat?sslmode=disable")
	v.SetDefault("postgres.max_open_conns", 25)
	v.SetDefault("postgres.max_idle_conns", 5)

	// Bind environment variables with KG_ prefix
	// e.g., KG_SERVER_PORT=8080, KG_GRPC_GRAPH_ENGINE_ADDR=localhost:50051
	v.SetEnvPrefix("KG")
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	v.AutomaticEnv()

	// Also bind specific env vars without prefix for Docker Compose compatibility
	_ = v.BindEnv("auth.jwt_secret", "JWT_SECRET")
	_ = v.BindEnv("redis.url", "REDIS_URL")
	_ = v.BindEnv("postgres.url", "POSTGRES_URL")
	_ = v.BindEnv("grpc.graph_engine_addr", "GRAPH_ENGINE_ADDR")
	_ = v.BindEnv("grpc.ai_pipeline_addr", "AI_PIPELINE_ADDR")

	// Try to read optional config file
	v.SetConfigName("config")
	v.SetConfigType("yaml")
	v.AddConfigPath(".")
	v.AddConfigPath("/etc/kgchat/")

	// Config file is optional — env vars are sufficient
	if err := v.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, fmt.Errorf("reading config file: %w", err)
		}
	}

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("unmarshaling config: %w", err)
	}

	return &cfg, nil
}
