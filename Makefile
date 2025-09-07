.PHONY: help rag test run clean install

# Default target
help:
	@echo "Available commands:"
	@echo "  make install  - Install dependencies"
	@echo "  make rag      - Build RAG index from PDFs"
	@echo "  make test     - Run all tests"
	@echo "  make run      - Start the server"
	@echo "  make clean    - Clean generated files"

# Install dependencies
install:
	pip install -r requirements.txt

# Build RAG index
rag:
	python scripts/build_rag.py

# Run tests
test:
	PYTHONPATH=. python -m pytest tests/ -v

# Start server
run:
	uvicorn app.main:app --reload --port 8000

# Clean generated files
clean:
	rm -rf rag/index/*
	rm -rf data/snapshot/*.json
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true