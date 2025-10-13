#!/bin/bash
# Neo4j setup script for GraphRAG

set -e

echo "🔧 Setting up Neo4j for GraphRAG..."

# Start Neo4j via Docker Compose
echo "Starting Neo4j container..."
docker-compose up -d neo4j

# Wait for Neo4j to be ready
echo "Waiting for Neo4j to be ready..."
sleep 10

# Check if Neo4j is running
if curl -s http://localhost:7474 > /dev/null; then
    echo "✅ Neo4j is running at http://localhost:7474"
else
    echo "❌ Neo4j failed to start. Check docker-compose logs neo4j"
    exit 1
fi

echo ""
echo "Neo4j credentials:"
echo "  URL: bolt://localhost:7687"
echo "  Username: neo4j"
echo "  Password: password"
echo "  Browser: http://localhost:7474"
echo ""
echo "Next: Run entity extraction to populate the graph"
echo "  python scripts/extract_entities.py"
