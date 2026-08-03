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

**Doc-only mode** is the key efficiency trick: the model runs **only at ingest time** to expand
documents. At **query time** we use a plain analyzer (`bert-uncased`) to tokenize the query — **no
model inference at search time**, so queries stay fast.

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

# 4. run the API and seed the sample corpus
make run                       # http://localhost:8000/docs  (leave running)
make seed                      # in another terminal
```

---

## Try it — compare the three modes

The sample corpus has horror, superhero, action, comedy, sci-fi, and animated films. A good query to
show the difference is one where the *concept* matters more than the exact words:

```bash
# Lexical only — needs the literal words to appear
curl -s 'http://localhost:8000/search?q=scary%20movies%20with%20ghosts&mode=lexical' | python3 -m json.tool

# Sparse semantic — matches the horror films via term expansion
curl -s 'http://localhost:8000/search?q=scary%20movies%20with%20ghosts&mode=sparse'  | python3 -m json.tool

# Hybrid — normalized blend of both (default)
curl -s 'http://localhost:8000/search?q=scary%20movies%20with%20ghosts&mode=hybrid'  | python3 -m json.tool
```

Or via POST:

```bash
curl -s -X POST http://localhost:8000/search \
  -H 'content-type: application/json' \
  -d '{"query": "space exploration", "mode": "hybrid", "size": 5}' | python3 -m json.tool
```

### Same queries in Dashboards Dev Tools (`http://localhost:5601` → Dev Tools)

```
POST hybrid-sparse-index/_search?search_pipeline=hybrid-sparse-search
{
  "_source": { "excludes": ["combined_text_embedding"] },
  "query": {
    "hybrid": {
      "queries": [
        { "match": { "combined_text": "scary movies with ghosts" } },
        { "neural_sparse": { "combined_text_embedding": {
            "query_text": "scary movies with ghosts", "analyzer": "bert-uncased" } } }
      ]
    }
  }
}
```

---

## What `make bootstrap` does (the 6 steps)

Implemented in [`scripts/bootstrap_opensearch.py`](scripts/bootstrap_opensearch.py), idempotent:

1. **ML Commons settings** — allow the model to run on this single (non-ML) node.
2. **Register + deploy** the sparse doc model; poll the task until `COMPLETED`, capture `model_id`
   (reuses an already-deployed model on re-run).
3. **Ingest pipeline** — `sparse_encoding` processor maps `combined_text → combined_text_embedding`.
4. **Index** — `combined_text` (`text`, BM25) + `combined_text_embedding` (`rank_features`, sparse),
   `default_pipeline` set, `1 shard / 0 replicas` (single-node → green health).
5. **Search pipeline** — `normalization-processor`, `min_max` + `arithmetic_mean`, weights
   `[lexical, sparse]` = `[0.3, 0.7]` (tune in `.env`).
6. **Persist** the `model_id` to `.env`.

To rebuild the index from scratch: `uv run python scripts/bootstrap_opensearch.py --recreate-index`.

---

## Project layout

```
opensearch/
├── docker/compose.yml              # OpenSearch (single-node) + Dashboards
├── scripts/bootstrap_opensearch.py # model + pipelines + index setup
├── app/
│   ├── config.py                   # settings from .env
│   ├── opensearch_client.py        # opensearch-py client (http, no auth)
│   ├── models.py                   # request/response schemas
│   ├── search.py                   # lexical / sparse / hybrid query builders
│   ├── ingest.py                   # bulk indexing
│   └── main.py                     # FastAPI routes
├── data/sample_docs.json           # sample movie corpus
└── tests/test_search.py            # smoke tests (skip if no cluster)
```

## API endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET`  | `/health` | Cluster status |
| `POST` | `/documents` | Index a list of `{name, combined_text}` docs |
| `POST` | `/documents/seed` | Index `data/sample_docs.json` |
| `POST` | `/search` | `{query, mode, size}` — `mode` ∈ `lexical\|sparse\|hybrid` |
| `GET`  | `/search` | `?q=...&mode=hybrid&size=5` |

Interactive docs at `http://localhost:8000/docs`.

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
- **`neural_sparse` analyzer error** — the doc-only `analyzer` query and the v3-distill model require
  a recent OpenSearch (this project pins `2.19.1`). If your image differs, update the tag in
  `docker/compose.yml` and `MODEL_*` in `.env`.
- **Cluster health `yellow`** — expected only if replicas > 0 on a single node; this setup uses
  0 replicas, so health should be `green`.
- **Slow seeding** — sparse encoding runs per document at ingest; the sample corpus is small on
  purpose.
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
