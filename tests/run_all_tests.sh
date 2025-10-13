#!/bin/bash
# Run All Tests Script for RAG Agent
# This script runs the complete test suite with all categories

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}RAG Agent - Complete Test Suite Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 1. Setup
echo -e "${YELLOW}📦 Step 1: Installing dependencies...${NC}"
pip install -r requirements.txt -q
pip install -r tests/requirements.txt -q
echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

# 2. Generate test data
echo -e "${YELLOW}🔧 Step 2: Generating test data...${NC}"
python tests/scripts/generate_test_data.py
python tests/fixtures/synthetic_corpus_generator.py
echo -e "${GREEN}✅ Test data generated${NC}"
echo ""

# 3. Create reports directory
mkdir -p reports
echo -e "${GREEN}✅ Reports directory created${NC}"
echo ""

# 4. Run tests by category
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Unit tests
echo -e "${YELLOW}🧪 Running Unit Tests...${NC}"
pytest tests/unit/ \
    -v \
    -m unit \
    --cov=core \
    --cov=app \
    --cov=api \
    --cov-report=xml:reports/coverage-unit.xml \
    --junitxml=reports/junit-unit.xml \
    -n auto || echo -e "${RED}⚠️  Some unit tests failed${NC}"
echo ""

# Integration tests
echo -e "${YELLOW}🔗 Running Integration Tests...${NC}"
pytest tests/integration/ \
    -v \
    -m integration \
    --junitxml=reports/junit-integration.xml || echo -e "${RED}⚠️  Some integration tests failed${NC}"
echo ""

# Behavioral tests
echo -e "${YELLOW}🎭 Running Behavioral Tests...${NC}"
pytest tests/behavior/ \
    -v \
    -m behavior \
    --junitxml=reports/junit-behavior.xml || echo -e "${RED}⚠️  Some behavioral tests failed${NC}"
echo ""

# Adversarial/Security tests
echo -e "${YELLOW}🛡️  Running Security Tests...${NC}"
pytest tests/adversarial/ \
    -v \
    -m adversarial \
    --junitxml=reports/junit-adversarial.xml || echo -e "${RED}⚠️  Some security tests failed${NC}"
echo ""

# Regression tests
echo -e "${YELLOW}🔄 Running Regression Tests...${NC}"
pytest tests/regression/ \
    -v \
    -m regression \
    --junitxml=reports/junit-regression.xml || echo -e "${RED}⚠️  Some regression tests failed${NC}"
echo ""

# Black-box tests (if API is running)
echo -e "${YELLOW}📦 Running Black-box API Tests...${NC}"
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    pytest tests/blackbox/ \
        -v \
        -m blackbox \
        --junitxml=reports/junit-blackbox.xml || echo -e "${RED}⚠️  Some black-box tests failed${NC}"
else
    echo -e "${YELLOW}⏭️  Skipping black-box tests (API not running at http://localhost:8000)${NC}"
fi
echo ""

# Performance tests (non-blocking)
echo -e "${YELLOW}⚡ Running Performance Tests...${NC}"
pytest tests/performance/ \
    -v \
    -m performance \
    --benchmark-only \
    --benchmark-json=reports/benchmark.json \
    --junitxml=reports/junit-performance.xml || echo -e "${YELLOW}⚠️  Performance tests completed with warnings${NC}"
echo ""

# 5. Full coverage report
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Generating Coverage Report${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

pytest \
    -v \
    -m "not performance" \
    --cov=core \
    --cov=app \
    --cov=api \
    --cov-report=xml:reports/coverage.xml \
    --cov-report=html:reports/coverage-html \
    --cov-report=term \
    --junitxml=reports/junit-full.xml \
    --cov-fail-under=85 \
    -n auto || echo -e "${RED}⚠️  Coverage below threshold or tests failed${NC}"
echo ""

# 6. Generate summary report
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Generating Test Report${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

python tests/scripts/test_reporter.py reports

# 7. Open HTML reports (optional)
if [ "$1" == "--open" ]; then
    echo ""
    echo -e "${YELLOW}📊 Opening HTML reports...${NC}"
    
    # macOS
    if command -v open &> /dev/null; then
        open reports/summary.html
        open reports/coverage-html/index.html
    # Linux
    elif command -v xdg-open &> /dev/null; then
        xdg-open reports/summary.html
        xdg-open reports/coverage-html/index.html
    # Windows (Git Bash)
    elif command -v start &> /dev/null; then
        start reports/summary.html
        start reports/coverage-html/index.html
    fi
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✨ Test suite execution complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "📊 Reports available at:"
echo -e "  • Summary:  ${BLUE}reports/summary.html${NC}"
echo -e "  • Coverage: ${BLUE}reports/coverage-html/index.html${NC}"
echo -e "  • JSON:     ${BLUE}reports/summary.json${NC}"
echo ""
echo -e "Run with ${YELLOW}--open${NC} flag to automatically open reports in browser"
echo ""
