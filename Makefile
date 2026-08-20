COMPOSE := podman-compose -f docker/compose.yml

.PHONY: help sync sync-offvpn up down logs sysctl bootstrap run prepare seed test lint

help:
	@echo "Targets:"
	@echo "  make sync         - install python deps with uv (on VPN / corporate index)"
	@echo "  make sync-offvpn  - install python deps from public PyPI (off VPN)"
	@echo "  make sysctl    - set vm.max_map_count in the podman VM (needed by OpenSearch)"
	@echo "  make up        - start OpenSearch + Dashboards (podman-compose)"
	@echo "  make down      - stop the stack"
	@echo "  make logs      - tail OpenSearch logs"
	@echo "  make bootstrap - register/deploy sparse model + create pipelines & index"
	@echo "  make run       - run the FastAPI app (http://localhost:8000/docs)"
	@echo "  make prepare   - build the Flickr caption dataset (data/flickr_docs.json)"
	@echo "  make seed      - index the dataset via the API"
	@echo "  make test      - run pytest"
	@echo "  make lint      - run ruff"

sync:
	uv sync --extra dev

# Off-VPN install: this Mac's shell points uv at a private package index (UV_DEFAULT_INDEX)
# and a corporate proxy/cert bundle, none of which resolve off VPN. Strip those for
# this one command and pull from public PyPI, using the macOS keychain (--native-tls).
sync-offvpn:
	env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
	    -u SSL_CERT_FILE -u CURL_CA_BUNDLE -u VIRTUAL_ENV \
	    UV_DEFAULT_INDEX=https://pypi.org/simple UV_INDEX= \
	    uv sync --extra dev --native-tls

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

prepare:
	uv run --no-sync python scripts/prepare_flickr.py

seed:
	curl -s -X POST http://localhost:8000/documents/seed | python3 -m json.tool

test:
	uv run --no-sync pytest

lint:
	uv run --no-sync ruff check .
