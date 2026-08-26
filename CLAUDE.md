# CLAUDE.md — project context for Claude Code

Read this first. It's the concise operating manual; deeper detail is in the linked docs.

## What this is
A **local, from-scratch OpenSearch "AI search" POC series** on a MacBook, wrapped in a
small FastAPI app. It walks OpenSearch's AI-search chapters one at a time, over **one
shared dataset** (Flickr30k image captions), so only the *technique* changes between
chapters — a controlled comparison. **Primary goal is learning**: the user wants to
genuinely understand how models/embeddings/AI search work, not just OpenSearch.

- Runtime: single-node **OpenSearch 3.7.0** + Dashboards via **podman-compose**, security off.
- Package manager **uv**; lint **ruff**; tests **pytest**; Python **3.12**.
- Repo: personal GitHub `anand4252/opensearch-poc`.

## Chapters — done vs next
- ✅ **Neural sparse** (doc-only, `rank_features`) + **Hybrid** (normalization pipeline) — `app/sparse/`
- ✅ **Dense semantic** (`knn_vector` + `neural`, msmarco model in-cluster) — `app/semantic/`
- ✅ **Multimodal** (CLIP in-app, **text→image** and **image→image**) — `app/multimodal/`
- ⏭️ **Next: Conversational RAG** — introduces **connectors** + an LLM (retrieve-then-generate)
- ⏭️ then **Agentic search** (builds on RAG), then **Dashboards AI-search workflows**

## Structure (per-technique packages + shared plumbing)
- `app/{sparse,semantic,multimodal}/` — each has `bootstrap.py`, `cli.py`, `search.py`, `routes.py`
  (multimodal adds `seed.py`, `clip_model.py`).
- Shared: `app/ml_commons.py` (model register/deploy/reuse + readiness), `app/deps.py`
  (settings, client, seed loader), `app/dataset.py` (build `flickr_docs.json` + `copy_images`),
  `app/ingest.py` (`bulk_index`, `_id`=image filename → idempotent re-seed), `app/config.py`,
  `app/models.py`, `app/main.py` (assembler + shared `/health`, `/dataset/prepare`).
- Endpoints grouped by tag: `/sparse/*`, `/semantic/*`, `/multimodal/*`, plus shared.
- CLIs run as modules: `python -m app.sparse.cli` etc. (Makefile wraps them).
- Data: `data/results.csv` **committed** (13 MB captions); `data/flickr_docs.json` and
  `data/images/` are **gitignored** (rebuilt / re-copied).

## Key design decisions (the "why")
- **One shared Flickr dataset** (captions for text chapters, images for multimodal). Doc shape
  `{name, combined_text}`: `name` = image filename (join key), `combined_text` = 5 captions joined.
- **Sparse/dense run the model INSIDE OpenSearch** (ingest pipeline + `neural`). **Multimodal runs
  CLIP in the APP** (OpenSearch is just the kNN store) — chosen because OpenSearch's *native*
  multimodal needs a **cloud** model (Bedrock/Cohere), which breaks the local/portable goal.
  The connector approach (local model behind an OpenSearch connector) was deferred — we'll learn
  **connectors in the RAG chapter** instead.
- Models: sparse = `opensearch-neural-sparse-encoding-doc-v3-distill`; dense =
  `msmarco-distilbert-base-tas-b` (768, l2); multimodal = `sentence-transformers/clip-ViT-B-32` (512, cosine).
- `multimodal` is an **optional dependency extra** (torch is heavy). Base app boots without it
  (lazy imports; the upload route is guarded on `python-multipart`).

## Gotchas (bit us before — check these)
- **Corporate proxy**  was set in the shell profile and unreachable
  off-VPN. For internet ops: `unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy` in that
  terminal, or use the `make *-offvpn` targets. **A different laptop may have a different network —
  re-check.**
- **PyCharm**: run as a FastAPI config using the **venv interpreter directly (not uv)**; no
  `--reload` when debugging; env vars go in the run config.
- **Hugging Face**: the newer **Xet** transfer fails behind the proxy → `HF_HUB_DISABLE_XET=1`
  (set in `app/multimodal/clip_model.py`). After the model is cached once, run with
  `HF_HUB_OFFLINE=1` to skip network update-checks.
- **ML models**: `register?deploy=true` can report COMPLETED before the model is inference-ready
  ("Model not ready yet") → `wait_until_ready` in `ml_commons`. `.plugins-ml-model` stores chunk
  docs; `find_deployed_model` excludes them (else it never detects an existing model).
- **git identity**: repo-local `user.email` is the GitHub privacy alias
  `22791991+anand4252@users.noreply.github.com` (global config stays the office identity).

## How to work with this user
- They're **learning AI/ML from the ground up** (embeddings, CLIP, RAG, weights, inference are new).
  **Explain concepts in plain English, with analogies**, before and after coding.
- **Plan before coding**; lay out trade-offs and give a clear recommendation.
- Prefer **conversational dialogue over multiple-choice** questions.
- They often like to **run commands themselves to learn** — provide the steps rather than always
  doing it for them.
- Frame skills honestly as hands-on POC experience (for their resume / next role).

## Pointers
- Setup / run: `README.md` (quick start, data setup, endpoints).
- Chapter deep-dives: `docs/dense-semantic-search.md`, `docs/multimodal-search.md`.
- Demo queries: `docs/test-scenarios.md`.
- Fuller decision narrative / handoff: `docs/PROJECT-JOURNAL.md`.
