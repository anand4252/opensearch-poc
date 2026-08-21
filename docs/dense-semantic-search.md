# Dense Semantic Search POC

The second chapter in the AI-search series: **dense** semantic search (text-embedding
model + `knn_vector` + `neural` query). It runs on the **same Flickr caption dataset**
as the neural-sparse POC — only the technique changes, so the two are a controlled
comparison.

Code lives in `app/semantic/`; it sits alongside `app/sparse/` (neural sparse) and
shares infra (`app/ml_commons.py`, `config.py`, `ingest.py`, `dataset.py`).

---

## How dense differs from neural sparse

| | Neural **sparse** (`app/sparse`) | **Dense** semantic (`app/semantic`) |
| --- | --- | --- |
| Index | `hybrid-sparse-index` | `semantic-dense-index` |
| Field type | `rank_features` (token→weight map) | `knn_vector` (fixed-length float array) |
| Model | `opensearch-neural-sparse-encoding-doc-v3-distill` | `msmarco-distilbert-base-tas-b` |
| Model output | sparse tokens | one 768-dim dense vector |
| Ingest processor | `sparse_encoding` | `text_embedding` |
| Query | `neural_sparse` / analyzer | `neural` (kNN / ANN) |
| **Query-time inference** | ❌ none (doc-only; analyzer tokenizes) | ✅ **yes** — the model encodes every query |
| Matching | lexical overlap + learned expansion | nearest-neighbor in vector space |

The defining contrast: dense semantic **runs the embedding model at query time** to turn
the query text into a vector, then does an ANN (HNSW) nearest-neighbor search. That's the
trade-off the OpenSearch "Semantic search" page is about — richer semantic matching, at
the cost of query-time model inference (the sparse doc-only POC has none).

## The model

**`huggingface/sentence-transformers/msmarco-distilbert-base-tas-b`** (v1.0.3,
`TORCH_SCRIPT`, **768-dim**).

This is what OpenSearch's own semantic-search tutorial uses, and the pretrained-models
reference tags it *"Optimized for semantic search."* It's trained for **asymmetric
query→passage retrieval** (MS MARCO), which fits our task exactly: a short search phrase
retrieving a matching caption.

## Configuration (`app/config.py` / `.env`)

```
DENSE_INDEX_NAME=semantic-dense-index
DENSE_INGEST_PIPELINE=semantic-dense-ingest
DENSE_MODEL_NAME=huggingface/sentence-transformers/msmarco-distilbert-base-tas-b
DENSE_MODEL_VERSION=1.0.3
DENSE_MODEL_FORMAT=TORCH_SCRIPT
DENSE_MODEL_ID=            # filled in by the dense bootstrap
EMBEDDING_DIMENSION=768
DENSE_SPACE_TYPE=l2
DENSE_ENGINE=lucene
KNN_FIELD=combined_text_knn
```

## What the bootstrap builds

`app/semantic/bootstrap.py` (idempotent — safe to re-run):

1. **ML settings** — loosen ML Commons guards for single-node local run (shared).
2. **Register + deploy** the dense model, reusing it if already deployed, then **wait
   until it actually serves inference** (see the readiness note below).
3. **Ingest pipeline** — `text_embedding` processor mapping `combined_text` → `combined_text_knn`:
   ```json
   { "text_embedding": { "model_id": "<id>", "field_map": { "combined_text": "combined_text_knn" } } }
   ```
4. **knn index** — `index.knn: true`, `default_pipeline` = the dense pipeline, and:
   ```json
   "combined_text_knn": {
     "type": "knn_vector",
     "dimension": 768,
     "method": { "name": "hnsw", "space_type": "l2", "engine": "lucene" }
   }
   ```
5. **Persist** `DENSE_MODEL_ID` into `.env`.

Pure dense semantic needs **no search pipeline** (that was only for sparse hybrid score
combination).

## The query

`app/semantic/search.py` issues a `neural` query against the vector field:

```json
GET semantic-dense-index/_search
{
  "size": 5,
  "_source": { "excludes": ["combined_text_knn"] },
  "query": {
    "neural": {
      "combined_text_knn": { "query_text": "puppy", "model_id": "<id>", "k": 5 }
    }
  }
}
```

`run_dense_search` resolves the model id from `DENSE_MODEL_ID`, falling back to the live
deployed model (`find_deployed_model`) — so search works right after a bootstrap without
restarting the app.

---

## Running it

```bash
make up                    # OpenSearch + Dashboards (if not already running)
make prepare               # build data/flickr_docs.json (shared dataset)
make bootstrap-semantic    # register/deploy model + dense pipeline & index (first run downloads ~250MB)
make run                   # start the API
make seed-semantic         # embed + index the dataset into the dense index
```

Then:

```bash
curl "localhost:8000/semantic/search?q=puppy"
```

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/semantic/bootstrap` | model + pipeline + knn index (`recreate_index`, `persist` flags) |
| POST | `/semantic/seed` | embed + index `data/flickr_docs.json` into the dense index |
| POST/GET | `/semantic/search` | dense `neural` (kNN) search; `?q=…&size=…` |

Seeding uses the image filename as `_id`, so re-seeding overwrites (no duplicates).

---

## Gotcha: "Model not ready yet"

A `register?deploy=true` task can report `COMPLETED` (and `model_state=DEPLOYED`) **before
the node has actually loaded the model for inference**. The first ingest/query then fails:

```
illegal_argument_exception: Model not ready yet. Please deploy the model first.
```

The bootstrap now guards against this: after deploy it calls `wait_until_ready`
(`app/ml_commons.py`), which probes a tiny `_predict`; if that fails it issues an explicit
`_deploy`, waits for that task, and polls `_predict` until it succeeds. So
`bootstrap-semantic` → `seed-semantic` no longer hits the race. (The same guard protects
the sparse bootstrap, since both share `register_and_deploy`.)

## Note on `l2` scores

With `space_type: l2`, scores are `1 / (1 + distance)` over unnormalized embeddings, so
they look small (≈ 0.02–0.03). That's expected — the **ranking** is what matters, not the
absolute score. Switch `DENSE_SPACE_TYPE` to `cosinesimil` if you prefer more intuitive
similarity numbers for a demo.

## Verified results

- Dense vector stored: `combined_text_knn` = **768 dims**.
- Seed: 1,000 docs embedded & indexed, **0 errors**.
- `puppy` (a word in **zero** captions) → dog photos ✅ — semantic, not string, matching.
- `cyclist` (3 literal captions) → bicycle riders/racers ✅.
- `a dog running on the beach`, `children playing outside` → relevant scenes ✅.
- `make test` → dense smoke tests pass (`tests/test_semantic.py`).

See `docs/test-scenarios.md` for the full demo query set (works for both POCs).

## Out of scope (later chapters)

Dense + lexical **hybrid** combination, and **multimodal** (embedding image pixels). This
chapter is pure dense semantic, mirroring the OpenSearch "Semantic search" page.
