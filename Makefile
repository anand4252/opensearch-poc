COMPOSE := podman-compose -f docker/compose.yml

.PHONY: help sync sync-offvpn sync-multimodal sync-multimodal-offvpn up down logs sysctl \
        bootstrap run prepare seed bootstrap-semantic seed-semantic \
        copy-images bootstrap-multimodal seed-multimodal test lint

# Override with the folder that holds the Flickr .jpg files, e.g.:
#   make copy-images IMAGES_SRC=~/Downloads/flickr30k_images/flickr30k_images
IMAGES_SRC ?= /path/to/flickr30k_images/flickr30k_images

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
	@echo "  make bootstrap-semantic - register/deploy dense model + create dense pipeline & index"
	@echo "  make seed-semantic      - index the dataset into the dense index via the API"
	@echo "  make sync-multimodal    - install the multimodal extra (CLIP/torch), on VPN"
	@echo "  make sync-multimodal-offvpn - same, from public PyPI (off VPN)"
	@echo "  make copy-images IMAGES_SRC=... - copy the subset .jpgs into data/images"
	@echo "  make bootstrap-multimodal - create the multimodal (CLIP) kNN index"
	@echo "  make seed-multimodal      - embed + index the subset images via the API"
	@echo "  make test      - run pytest"
	@echo "  make lint      - run ruff"

sync:
	uv sync --extra dev

# Off-VPN install: this Mac's shell points uv at Artifactory (UV_DEFAULT_INDEX)
# and a corporate proxy/cert bundle, none of which resolve off VPN. Strip those for
# this one command and pull from public PyPI, using the macOS keychain (--native-tls).
sync-offvpn:
	env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
	    -u SSL_CERT_FILE -u CURL_CA_BUNDLE -u VIRTUAL_ENV \
	    UV_DEFAULT_INDEX=https://pypi.org/simple UV_INDEX= \
	    uv sync --extra dev --native-tls

# Multimodal chapter needs the heavy `multimodal` extra (sentence-transformers + torch).
sync-multimodal:
	uv sync --extra dev --extra multimodal

sync-multimodal-offvpn:
	env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
	    -u SSL_CERT_FILE -u CURL_CA_BUNDLE -u VIRTUAL_ENV \
	    UV_DEFAULT_INDEX=https://pypi.org/simple UV_INDEX= \
	    uv sync --extra dev --extra multimodal --native-tls

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
	uv run --no-sync python -m app.sparse.cli

run:
	uv run --no-sync uvicorn app.main:app --reload

prepare:
	uv run --no-sync python scripts/prepare_flickr.py

seed:
	curl -s -X POST http://localhost:8000/sparse/seed | python3 -m json.tool

bootstrap-semantic:
	uv run --no-sync python -m app.semantic.cli

seed-semantic:
	curl -s -X POST http://localhost:8000/semantic/seed | python3 -m json.tool

copy-images:
	uv run --no-sync python scripts/prepare_flickr.py --copy-images --images-src $(IMAGES_SRC)

bootstrap-multimodal:
	uv run --no-sync python -m app.multimodal.cli

seed-multimodal:
	curl -s -X POST http://localhost:8000/multimodal/seed | python3 -m json.tool

test:
	uv run --no-sync pytest

lint:
	uv run --no-sync ruff check .
