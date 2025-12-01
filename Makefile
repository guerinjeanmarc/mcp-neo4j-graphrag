.PHONY: install dev test format lint clean run

# Install the package
install:
	uv pip install -e .

# Install with dev dependencies
dev:
	uv pip install -e ".[dev]"

# Run tests (placeholder)
test:
	@echo "Tests not yet implemented"

# Format code
format:
	ruff format src/

# Lint code
lint:
	ruff check src/

# Type check
typecheck:
	pyright src/

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

# Run the server locally
run:
	mcp-neo4j-graphrag --transport stdio

# Build package
build:
	uv build

