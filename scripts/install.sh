#!/bin/bash
# Installation script for Production RAG System

set -e

echo "🚀 Installing Production RAG System Dependencies..."

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install -r requirements.txt

# Install LangGraph & agent frameworks
echo "📦 Installing LangGraph and agent frameworks..."
pip install -r requirements-langgraph.txt

# Install rank-bm25
echo "📦 Installing additional retrieval libraries..."
pip install rank-bm25 pypdf python-docx

# Verify installations
echo "✅ Verifying installations..."
python -c "import fastapi; print('✓ FastAPI')"
python -c "import langgraph; print('✓ LangGraph')"
python -c "import langchain; print('✓ LangChain')"
python -c "from sentence_transformers import CrossEncoder; print('✓ Sentence Transformers')"
python -c "import neo4j; print('✓ Neo4j Driver')"
python -c "from rank_bm25 import BM25Okapi; print('✓ BM25')"

# Optional: Install AutoGen and CrewAI if not already installed
echo "📦 Installing multi-agent frameworks..."
pip install pyautogen crewai crewai-tools || echo "⚠️  Multi-agent frameworks installation failed (optional)"

echo ""
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Start Neo4j: docker-compose up -d neo4j"
echo "2. Configure: cp configs/default.yaml configs/local.yaml"
echo "3. Seed data: python seeds/seed_data.py"
echo "4. Run tests: pytest tests/ -v"
echo "5. Start API: uvicorn api.main:app --reload"
echo ""
