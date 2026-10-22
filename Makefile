.PHONY: help install dev test lint format typecheck check docker-build docker-up docker-down db-up db-migrate db-revision db-downgrade

help:  ## Show this help menu
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies into virtual environment
	uv sync --all-extras

dev:  ## Run local development server with auto-reload
	uv run uvicorn ragforge.main:app --reload --host 0.0.0.0 --port 8000

test:  ## Run pytest test suite
	uv run pytest -v

lint:  ## Run ruff linter
	uv run ruff check .

format:  ## Format code with ruff
	uv run ruff format .

typecheck:  ## Run static type checking with mypy
	uv run mypy src tests

check: lint format typecheck test  ## Run all static checks and tests

db-up:  ## Start only the PostgreSQL database container
	docker compose up -d postgres

db-migrate:  ## Apply all pending Alembic database migrations
	uv run alembic upgrade head

db-revision:  ## Autogenerate a new Alembic migration revision from models (e.g. make db-revision m="add_tables")
	uv run alembic revision --autogenerate -m "$(or $(m),schema_update)"

db-downgrade:  ## Rollback the last applied migration
	uv run alembic downgrade -1

docker-build:  ## Build production Docker image
	docker build -t ragforge:latest .

docker-up:  ## Start application and PostgreSQL using Docker Compose
	docker compose up -d --build

docker-down:  ## Stop and remove Docker Compose containers
	docker compose down
