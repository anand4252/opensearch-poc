# Local OpenSearch Hybrid Search (BM25 + Neural Sparse)

A from-scratch, reproducible local setup for **hybrid search** on OpenSearch, wrapped in a small
FastAPI service. It mirrors an AWS OpenSearch setup: **BM25 lexical** search combined with
**neural sparse** semantic search (in **doc-only mode**), fused by a `normalization-processor`
search pipeline.

- **Runtime:** single-node OpenSearch + Dashboards via `podman-compose`
- **Semantic model:** `amazon/neural-sparse/opensearch-neural-sparse-encoding-doc-v3-distill`,
  deployed *inside* the cluster (ML Commons)
- **API:** FastAPI with `lexical` / `sparse` / `hybrid` search modes
- **Package manager:** `uv`

## Docs / AI-search chapters

This repo is a series of AI-search POCs over one shared Flickr caption dataset — only the
technique changes between chapters.

| Chapter | Technique | Code | Doc |
| --- | --- | --- | --- |
| Neural sparse + hybrid | `rank_features`, doc-only | `app/sparse/` | this README |
| Dense semantic | `knn_vector` + `neural` query | `app/semantic/` | [docs/dense-semantic-search.md](docs/dense-semantic-search.md) |

- **[docs/test-scenarios.md](docs/test-scenarios.md)** — demo query set (works for both chapters).

---

## How hybrid search works here

| Layer | What it does | OpenSearch feature |
| --- | --- | --- |
| **Lexical (BM25)** | Exact keyword / term matching | `match` query on a `text` field |
| **Semantic (neural sparse)** | Expands text into a map of `token → weight` so related terms match even without the exact word | `sparse_encoding` ingest processor → `rank_features` field → `neural_sparse` query |
| **Fusion** | Normalizes the two score distributions and combines them with weights | `normalization-processor` search pipeline (`min_max` + `arithmetic_mean`) |

**Neural sparse vs dense (kNN):** dense search embeds text into a fixed-length float vector and
compares with kNN. *Sparse* search instead produces a sparse `token → weight` map (like an expanded
bag of words). It stores into a Lucene `rank_features` field — no vector index — and tends to be
strong on keyword-ish relevance while still capturing term expansion (synonyms, related concepts).

**Doc-only mode** is the key efficiency trick: the encoding model runs **only at ingest time** to
expand documents. At **query time** a built-in analyzer (`bert-uncased`) tokenizes the query — **no
model inference at search time**, so queries stay fast.

> **Version note:** this project targets **OpenSearch 3.x** (pinned `3.7.0`), where the `neural_sparse`
> query accepts an `analyzer` field directly — no separate query-side model to deploy. (On older 2.x
> you had to deploy a tokenizer model and reference it by `model_id`; 3.x removed that extra step.)

---

## Prerequisites

- macOS with **podman** + **podman-compose**, and **uv** installed
- The podman machine needs memory headroom. Recommended (host permitting):

```bash
podman machine stop
podman machine set --memory 8192 --cpus 4
podman machine start
podman machine list          # confirm it now shows 8GiB
```

- OpenSearch requires a high `vm.max_map_count` **inside the podman VM**. This resets when the VM
  restarts, so re-run it after any `podman machine start`:

```bash
podman machine ssh 'sudo sysctl -w vm.max_map_count=262144'
# or: make sysctl
```

---

## Quick start

```bash
# 0. one-time: kernel setting inside the podman VM
make sysctl

# 1. start OpenSearch + Dashboards
make up
#   wait ~1 min, then verify:
curl -s http://localhost:9200/_cluster/health | python3 -m json.tool
#   Dashboards UI: http://localhost:5601

# 2. install python deps and create your .env
make sync
cp .env.example .env

# 3. register/deploy the sparse model + create pipelines & index
#    (writes the resolved MODEL_ID back into .env)
make bootstrap

# 4. build the dataset from the committed captions (data/results.csv -> data/flickr_docs.json)
make prepare

# 5. run the API and seed the dataset
make run                       # http://localhost:8000/docs  (leave running)
make seed                      # in another terminal
```

---

## Try it — compare the three modes

The dataset is **Flickr30k image captions** — each document is one image, its five human-written
captions merged into `combined_text`, and `name` is the image filename. So a search is really
*text → image retrieval*: type a scene, get back the image whose captions best match. Good queries
are scene descriptions where the *concept* matters more than the exact words:

```bash
# Lexical only — needs the literal words to appear
curl -s 'http://localhost:8000/sparse/search?q=a%20dog%20running%20on%20the%20beach&mode=lexical' | python3 -m json.tool

# Sparse semantic — also matches captions via term expansion
curl -s 'http://localhost:8000/sparse/search?q=a%20dog%20running%20on%20the%20beach&mode=sparse'  | python3 -m json.tool

# Hybrid — normalized blend of both (default)
curl -s 'http://localhost:8000/sparse/search?q=a%20dog%20running%20on%20the%20beach&mode=hybrid'  | python3 -m json.tool
```

Or via POST:

```bash
curl -s -X POST http://localhost:8000/sparse/search \
  -H 'content-type: application/json' \
  -d '{"query": "children playing outside", "mode": "hybrid", "size": 5}' | python3 -m json.tool
```

Each hit's `name` (e.g. `1057251835.jpg`) is the matching image — look it up in the Flickr images
if you've fetched them (see **Data setup**).

### Same queries in Dashboards Dev Tools (`http://localhost:5601` → Dev Tools)

```
POST hybrid-sparse-index/_search?search_pipeline=hybrid-sparse-search
{
  "_source": { "excludes": ["combined_text_embedding"] },
  "query": {
    "hybrid": {
      "queries": [
        { "match": { "combined_text": "a dog running on the beach" } },
        { "neural_sparse": { "combined_text_embedding": {
            "query_text": "a dog running on the beach", "analyzer": "bert-uncased" } } }
      ]
    }
  }
}
```

---

## Data setup

The dataset is the **[Flickr30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)**
captions — shared across the whole "AI search" POC series (neural sparse → dense semantic →
multimodal), so the *data* stays constant and only the *technique* changes.

**What's in the repo vs not:**

| Asset | Size | In git? | Notes |
| --- | --- | --- | --- |
| `data/results.csv` | 13 MB | ✅ committed | the caption source of truth; all text POCs need only this |
| `data/flickr_docs.json` | <1 MB | ❌ gitignored | built from the CSV by `make prepare` (deterministic) |
| the 31k `.jpg` images | 8.5 GB | ❌ never | only the **multimodal** POC needs pixels (see below) |

`make prepare` groups the 5 captions per image, takes a deterministic subset (sorted, first *N*),
and writes `data/flickr_docs.json`. Tune the size in `.env` (`FLICKR_SUBSET_SIZE`, default `1000`) or
per-run: `uv run python scripts/prepare_flickr.py --size 500`.

**Reproduce on a new machine** — the text POCs need **no image download**:

```bash
git clone <repo> && cd opensearch-poc
make up && make sync && make bootstrap && make prepare && make seed
```

**Images (only for the future multimodal POC).** The 8.5 GB of images aren't committed. When you get
to multimodal, rehydrate just the subset one of two ways:

- **Re-download from Kaggle**, then copy the pinned subset into `data/images/` (gitignored):
  ```bash
  uv run python scripts/prepare_flickr.py --copy-images \
      --images-src ~/Downloads/flickr30k_images/flickr30k_images
  ```
- **Git LFS** the ~250 MB subset instead, for a fully self-contained `git clone` (no Kaggle account),
  at the cost of LFS quota.

---

## What `make bootstrap` does (the 6 steps)

Logic in [`app/sparse/bootstrap.py`](app/sparse/bootstrap.py), driven by the CLI
[`app/sparse/cli.py`](app/sparse/cli.py) (`make bootstrap`), idempotent:

1. **ML Commons settings** — allow the model to run on this single (non-ML) node.
2. **Register + deploy** the sparse ingest model; poll the task until `COMPLETED`, capture `MODEL_ID`
   (reuses an already-deployed model on re-run). No query-side model needed on 3.x.
3. **Ingest pipeline** — `sparse_encoding` processor maps `combined_text → combined_text_embedding`.
4. **Index** — `combined_text` (`text`, BM25) + `combined_text_embedding` (`rank_features`, sparse),
   `default_pipeline` set, `1 shard / 0 replicas` (single-node → green health).
5. **Search pipeline** — `normalization-processor`, `min_max` + `arithmetic_mean`, weights
   `[lexical, sparse]` = `[0.3, 0.7]` (tune in `.env`).
6. **Persist** the `model_id` to `.env`.

To rebuild the index from scratch: `uv run python -m app.sparse.cli --recreate-index`.

---

## Project layout

```
opensearch-poc/
├── docker/compose.yml              # OpenSearch (single-node) + Dashboards
├── scripts/
│   └── prepare_flickr.py           # build the shared caption dataset from data/results.csv
├── app/
│   ├── config.py                   # settings from .env
│   ├── opensearch_client.py        # opensearch-py client (http, no auth)
│   ├── models.py                   # request/response schemas
│   ├── ingest.py                   # bulk indexing (shared)
│   ├── dataset.py                  # build flickr_docs.json from captions (shared)
│   ├── ml_commons.py               # model register/deploy/reuse + readiness (shared)
│   ├── deps.py                     # shared FastAPI request deps (settings, client, seed loader)
│   ├── main.py                     # app assembler: shared endpoints + mounts routers
│   ├── sparse/                     # neural-sparse POC (BM25 + sparse, hybrid)
│   │   ├── bootstrap.py            #   pipelines + index wiring
│   │   ├── cli.py                  #   `python -m app.sparse.cli` (make bootstrap)
│   │   ├── search.py               #   lexical / sparse / hybrid query builders
│   │   └── routes.py               #   /sparse/* endpoints
│   └── semantic/                   # dense semantic POC (knn_vector + neural)
│       ├── bootstrap.py            #   text_embedding pipeline + knn index wiring
│       ├── cli.py                  #   `python -m app.semantic.cli` (make bootstrap-semantic)
│       ├── search.py               #   neural (kNN) query
│       └── routes.py               #   /semantic/* endpoints
├── data/
│   ├── results.csv                 # Flickr30k captions (committed, 13 MB)
│   ├── flickr_docs.json            # built by `make prepare` (gitignored)
│   └── sample_docs.json            # small movie dataset (alternate seed source)
├── docs/                           # per-chapter write-ups + test scenarios
└── tests/                          # smoke tests (skip if no cluster)
```

## API endpoints

Endpoints are grouped by tag in Swagger: **shared**, **sparse**, **semantic**.

| Tag | Method | Path | Purpose |
| --- | --- | --- | --- |
| shared | `GET`  | `/health` | Cluster status |
| shared | `POST` | `/dataset/prepare` | Build the shared dataset from `data/results.csv` (optional `?size=`); same as `make prepare` |
| sparse | `POST` | `/sparse/bootstrap` | ML settings + sparse model + pipelines + index (`?recreate_index=&persist=`) |
| sparse | `POST` | `/sparse/documents` | Ad-hoc index a list of `{name, combined_text}` docs |
| sparse | `POST` | `/sparse/seed` | Index the prepared dataset into the sparse index |
| sparse | `POST` | `/sparse/search` | `{query, mode, size}` — `mode` ∈ `lexical\|sparse\|hybrid` |
| sparse | `GET`  | `/sparse/search` | `?q=...&mode=hybrid&size=5` |
| semantic | `POST` | `/semantic/bootstrap` | Dense model + text_embedding pipeline + knn index |
| semantic | `POST` | `/semantic/seed` | Index the prepared dataset into the dense index |
| semantic | `POST` | `/semantic/search` | Dense `neural` (kNN) search; `{query, size}` or `?q=...` |
| semantic | `GET`  | `/semantic/search` | `?q=puppy&size=5` |

Interactive docs at `http://localhost:8000/docs`.

**Demo entirely from Swagger** (no terminal): `make run`, open `/docs`, then click
**`POST /dataset/prepare`** → **`POST /sparse/seed`** → **`POST /sparse/search`** (or the
`/semantic/*` equivalents). (`prepare` writes the dataset file locally, so it's a convenience
for local demos, not a production endpoint.)

---

## Relationship to the AWS setup

This reproduces an AWS OpenSearch (`hackathonv2`) hybrid-sparse configuration. Local differences:

| | AWS | Local |
| --- | --- | --- |
| Auth | SigV4 (`AWS4Auth`) over HTTPS | plain HTTP, security plugin disabled |
| Shards/replicas | 3 shards / 2 replicas | 1 shard / 0 replicas |
| Runtime | managed domain | single-node podman container |

The **model, ingest pipeline, index mapping, `neural_sparse` doc-only query, and normalization
pipeline are identical** in shape.

---

## Troubleshooting

- **`max virtual memory areas vm.max_map_count [65530] is too low`** — run `make sysctl` (it resets
  when the podman VM restarts).
- **OpenSearch container exits / OOM** — give the podman machine more memory (see Prerequisites).
- **Model stuck / `FAILED` during bootstrap** — check ML logs: `make logs`. Ensure the ML Commons
  settings applied and the node has memory headroom. Re-running `make bootstrap` reuses a deployed
  model.
- **`neural_sparse` analyzer error** (`does not support [analyzer] field`) — the `analyzer` form
  needs **OpenSearch 3.x** (this project pins `3.7.0`). On older 2.x, deploy the tokenizer model
  `opensearch-neural-sparse-tokenizer-v1` and pass its `model_id` instead of `analyzer`.
- **Cluster health `yellow`** — expected only if replicas > 0 on a single node; this setup uses
  0 replicas, so health should be `green`.
- **Slow seeding** — sparse encoding runs the model per document at ingest, so seed time scales with
  `FLICKR_SUBSET_SIZE` (~1 min for 1,000 on a single CPU node). Lower it for faster iteration:
  `uv run python scripts/prepare_flickr.py --size 500`, then re-seed.
- **`make sync` can't reach the package index** — if `uv` fails with a DNS/connection error to a
  corporate Artifactory (or a `UnknownIssuer` TLS error), you're likely off-VPN. Either connect to
  VPN, or install once from public PyPI:
  ```bash
  env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u SSL_CERT_FILE -u CURL_CA_BUNDLE \
    UV_DEFAULT_INDEX=https://pypi.org/simple UV_INDEX= uv sync --extra dev --native-tls
  ```
  After deps are installed, `make run` / `make test` use `uv run --no-sync`, so they work without
  hitting the index again.

## Manage the stack

```bash
make down     # stop containers (data volume persists)
make logs     # tail OpenSearch logs
make test     # run pytest (integration tests skip if cluster is down)
make lint     # ruff
```
