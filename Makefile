.PHONY: help install dev test lint format typecheck check docker-build docker-up docker-down

help:  ## Show this help menu
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies into virtual environment
	uv venv --python 3.12
	uv pip install -e ".[dev]"
	uv lock

dev:  ## Run local development server with auto-reload
	.venv/bin/uvicorn ragforge.main:app --reload --host 0.0.0.0 --port 8000

test:  ## Run pytest test suite
	.venv/bin/pytest -v

lint:  ## Run ruff linter
	.venv/bin/ruff check .

format:  ## Format code with ruff
	.venv/bin/ruff format .

typecheck:  ## Run static type checking with mypy
	.venv/bin/mypy src tests

check: lint typecheck test  ## Run all static checks and tests

docker-build:  ## Build production Docker image
	docker build -t ragforge:latest .

docker-up:  ## Start application and PostgreSQL using Docker Compose
	docker compose up -d --build

docker-down:  ## Stop and remove Docker Compose containers
	docker compose down
