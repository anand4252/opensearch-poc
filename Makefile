COMPOSE := podman-compose -f docker/compose.yml

.PHONY: help sync up down logs sysctl bootstrap run seed test lint

help:
	@echo "Targets:"
	@echo "  make sync      - install python deps with uv"
	@echo "  make sysctl    - set vm.max_map_count in the podman VM (needed by OpenSearch)"
	@echo "  make up        - start OpenSearch + Dashboards (podman-compose)"
	@echo "  make down      - stop the stack"
	@echo "  make logs      - tail OpenSearch logs"
	@echo "  make bootstrap - register/deploy sparse model + create pipelines & index"
	@echo "  make run       - run the FastAPI app (http://localhost:8000/docs)"
	@echo "  make seed      - index the sample corpus via the API"
	@echo "  make test      - run pytest"
	@echo "  make lint      - run ruff"

sync:
	uv sync --extra dev

# sysctl:
# 	podman machine ssh 'sudo sysctl -w vm.max_map_count=262144'

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f opensearch-node1

# --no-sync runs from the existing venv without re-resolving against the package
# index — handy when the corporate index/VPN is unreachable. Run `make sync` first.
bootstrap:
	uv run --no-sync python scripts/bootstrap_opensearch.py

run:
	uv run --no-sync uvicorn app.main:app --reload

seed:
	curl -s -X POST http://localhost:8000/documents/seed | python3 -m json.tool

test:
	uv run --no-sync pytest

lint:
	uv run --no-sync ruff check .
