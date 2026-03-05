// Package ws provides a WebSocket hub for real-time graph updates.
// Uses the gorilla/websocket library with a goroutine-based fan-out pattern.
//
// Architecture:
// - Hub runs as a single goroutine managing all client connections
// - Each Client has a dedicated write goroutine reading from its send channel
// - Graph updates can be targeted to specific sessions or broadcast globally
package ws

import (
	"encoding/json"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

const (
	// writeWait is the max time to wait for a write to complete.
	writeWait = 10 * time.Second
	// pongWait is the max time to wait for a pong response.
	pongWait = 60 * time.Second
	// pingPeriod is how often we send pings (must be < pongWait).
	pingPeriod = (pongWait * 9) / 10
	// maxMessageSize is the maximum incoming WebSocket message size.
	maxMessageSize = 512 * 1024 // 512KB
)

// upgrader configures the WebSocket upgrade from HTTP.
var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 4096,
	// Allow all origins in dev; restrict in production via config
	CheckOrigin: func(r *http.Request) bool { return true },
}

// GraphUpdate represents a real-time graph change pushed to clients.
type GraphUpdate struct {
	Type      string          `json:"type"`       // "node_added", "edge_added", "subgraph"
	SessionID string          `json:"session_id"` // Target session (empty = broadcast)
	Payload   json.RawMessage `json:"payload"`
}

// Client represents a single WebSocket connection with its session context.
type Client struct {
	hub       *Hub
	conn      *websocket.Conn
	send      chan []byte
	sessionID string
	userID    string
}

// Hub maintains the set of active WebSocket clients and handles fan-out.
type Hub struct {
	// clients maps each connected client to true.
	clients map[*Client]bool
	// sessionClients maps session IDs to their connected clients.
	sessionClients map[string]map[*Client]bool
	// broadcast receives messages to send to all clients.
	broadcast chan []byte
	// register receives new client connections.
	register chan *Client
	// unregister receives client disconnections.
	unregister chan *Client
	// mu protects the clients and sessionClients maps.
	mu     sync.RWMutex
	logger *zap.Logger
}

// NewHub creates a new WebSocket hub.
func NewHub(logger *zap.Logger) *Hub {
	return &Hub{
		clients:        make(map[*Client]bool),
		sessionClients: make(map[string]map[*Client]bool),
		broadcast:      make(chan []byte, 256),
		register:       make(chan *Client),
		unregister:     make(chan *Client),
		logger:         logger,
	}
}

// Run starts the hub's main event loop as a goroutine.
// This is the only goroutine that mutates the clients maps.
func (h *Hub) Run() {
	for {
		select {
		case client := <-h.register:
			h.mu.Lock()
			h.clients[client] = true
			// Track client by session for targeted updates
			if client.sessionID != "" {
				if _, ok := h.sessionClients[client.sessionID]; !ok {
					h.sessionClients[client.sessionID] = make(map[*Client]bool)
				}
				h.sessionClients[client.sessionID][client] = true
			}
			h.mu.Unlock()

			h.logger.Info("WebSocket client connected",
				zap.String("session_id", client.sessionID),
				zap.Int("total_clients", len(h.clients)),
			)

		case client := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[client]; ok {
				delete(h.clients, client)
				close(client.send)
				// Remove from session tracking
				if client.sessionID != "" {
					if sessions, ok := h.sessionClients[client.sessionID]; ok {
						delete(sessions, client)
						if len(sessions) == 0 {
							delete(h.sessionClients, client.sessionID)
						}
					}
				}
			}
			h.mu.Unlock()

			h.logger.Info("WebSocket client disconnected",
				zap.String("session_id", client.sessionID),
				zap.Int("total_clients", len(h.clients)),
			)

		case message := <-h.broadcast:
			h.mu.RLock()
			for client := range h.clients {
				select {
				case client.send <- message:
				default:
					// Client send buffer full — disconnect
					close(client.send)
					delete(h.clients, client)
				}
			}
			h.mu.RUnlock()
		}
	}
}

// BroadcastGraphUpdate sends a graph update to clients in a specific session,
// or to all clients if sessionID is empty.
func (h *Hub) BroadcastGraphUpdate(update GraphUpdate) {
	data, err := json.Marshal(update)
	if err != nil {
		h.logger.Error("Failed to marshal graph update", zap.Error(err))
		return
	}

	if update.SessionID == "" {
		// Broadcast to all clients
		h.broadcast <- data
		return
	}

	// Send to specific session's clients only
	h.mu.RLock()
	defer h.mu.RUnlock()

	if clients, ok := h.sessionClients[update.SessionID]; ok {
		for client := range clients {
			select {
			case client.send <- data:
			default:
				h.logger.Warn("Client send buffer full, skipping",
					zap.String("session_id", update.SessionID),
				)
			}
		}
	}
}

// ServeWS handles WebSocket upgrade requests and registers the new client.
func (h *Hub) ServeWS(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		h.logger.Error("WebSocket upgrade failed", zap.Error(err))
		return
	}

	// Extract session and user IDs from query params or context
	sessionID := r.URL.Query().Get("session_id")
	userID := r.URL.Query().Get("user_id")

	client := &Client{
		hub:       h,
		conn:      conn,
		send:      make(chan []byte, 256),
		sessionID: sessionID,
		userID:    userID,
	}

	h.register <- client

	// Start the client's read and write pumps as separate goroutines
	go client.writePump()
	go client.readPump()
}

// readPump reads incoming messages from the WebSocket connection.
// Currently only used for ping/pong keepalive; clients don't send data.
func (c *Client) readPump() {
	defer func() {
		c.hub.unregister <- c
		c.conn.Close()
	}()

	c.conn.SetReadLimit(maxMessageSize)
	c.conn.SetReadDeadline(time.Now().Add(pongWait))
	c.conn.SetPongHandler(func(string) error {
		c.conn.SetReadDeadline(time.Now().Add(pongWait))
		return nil
	})

	for {
		_, _, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseNormalClosure) {
				c.hub.logger.Warn("WebSocket read error",
					zap.Error(err),
					zap.String("session_id", c.sessionID),
				)
			}
			break
		}
	}
}

// writePump sends messages from the send channel to the WebSocket connection.
// Also sends periodic pings for keepalive.
func (c *Client) writePump() {
	ticker := time.NewTicker(pingPeriod)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()

	for {
		select {
		case message, ok := <-c.send:
			c.conn.SetWriteDeadline(time.Now().Add(writeWait))

			if !ok {
				// Hub closed the channel — send close frame
				c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			w, err := c.conn.NextWriter(websocket.TextMessage)
			if err != nil {
				return
			}
			w.Write(message)

			// Drain any queued messages to batch writes
			n := len(c.send)
			for i := 0; i < n; i++ {
				w.Write([]byte("\n"))
				w.Write(<-c.send)
			}

			if err := w.Close(); err != nil {
				return
			}

		case <-ticker.C:
			c.conn.SetWriteDeadline(time.Now().Add(writeWait))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}
