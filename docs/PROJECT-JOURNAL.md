# Project Journal

A running narrative of *why* this project is shaped the way it is — the decisions, the
dead-ends, and the plan. `CLAUDE.md` has the concise version; this is the story behind it.
(As of 2026-08.) Delete this and its link in `CLAUDE.md` once the series is complete — by
then the per-chapter `docs/*.md` are the permanent record.

## The goal
Learn OpenSearch's "AI search" features **and**, more importantly, understand how the
underlying models/AI actually work — enough to speak to them confidently and pick up a real
project. Each of OpenSearch's AI-search chapters is built as its own POC, but over **one
shared dataset** so only the *technique* varies. This makes it a controlled comparison and
keeps the learning focused on the technique, not the data.

## Why Flickr30k as the one dataset
It carries **both modalities**: natural-language **captions** (for the text chapters —
sparse, dense) and the **images** themselves (for multimodal). Joined by `image_name`. So
the same 1,000-image subset flows through every chapter. `results.csv` (13 MB captions) is
committed; the 8.5 GB of images are not (re-fetch from Kaggle; copy the subset locally).

Doc shape is `{name, combined_text}`: `name` = image filename (the join key reused
everywhere), `combined_text` = the 5 human captions joined. Indexing uses `_id` = filename,
so re-seeding overwrites instead of duplicating.

## Chapter-by-chapter

### Neural sparse + hybrid (done)
Doc-only mode: a `sparse_encoding` model expands each doc into token→weight `rank_features`
at ingest; the query side uses a built-in analyzer (no query-time model). Hybrid combines
BM25 + sparse via a `normalization-processor` search pipeline. This was the original POC;
the shared-dataset idea came after, when the 18-doc movie sample proved too small.

### Dense semantic (done)
Text-embedding model (`msmarco-distilbert-base-tas-b`, 768-dim, chosen because OpenSearch's
own tutorial uses it and it's tuned for query→passage retrieval) + `knn_vector` + `neural`
query. Key contrast with sparse: dense **runs the model at query time**. Model runs *inside*
OpenSearch (ingest pipeline + `neural`). See `docs/dense-semantic-search.md`.

### Multimodal (done) — the Option A/B/C decision
The big fork. OpenSearch's *native* multimodal (`text_image_embedding` + `neural`) requires
an **external cloud model** (Bedrock Titan / Cohere) — no local multimodal model exists in
ml-commons (its pretrained list is text-only). That conflicts with the fully-local,
rebuild-from-clone goal. Options considered:
- **A. Cloud connector** (Bedrock/Cohere) — faithful to the docs, but needs an account/keys. ❌
- **B. CLIP in our app** — we embed images/text with CLIP locally; OpenSearch is just the kNN
  store (plain `knn` query, we pass the vector). ✅ **Chosen.**
- **C. Local CLIP behind an OpenSearch connector** — faithful native pipeline, fully local,
  and teaches connectors — but more setup (a CLIP service + connector blueprint).

We chose **B** because the user's goal is understanding *models/AI*, and B keeps the model
front-and-center in code they own. **Connectors (the thing C would have taught) are deferred
to the RAG chapter**, where they matter more (connecting an LLM). CLIP = `clip-ViT-B-32`
(512-dim, cosine): one model, one shared space for images and text. Includes **text→image**
and **image→image** (upload, or by an indexed image name). See `docs/multimodal-search.md`.

## Architecture decisions
- **Per-technique packages** (`app/sparse`, `app/semantic`, `app/multimodal`) so each chapter
  is a self-contained, readable unit; genuinely-shared plumbing was extracted to
  `app/ml_commons.py`, `app/deps.py`, `app/dataset.py`, `app/ingest.py`.
- **Endpoints grouped by prefix/tag** (`/sparse/*`, `/semantic/*`, `/multimodal/*`) so Swagger
  reads clearly; shared steps are `/health` and `/dataset/prepare`. `main.py` is a thin
  assembler that mounts the routers.
- **CLIs as modules** (`python -m app.<tech>.cli`) — no `sys.path` hacks.
- Every "do something" step is a **POST** (prepare, copy-images, bootstrap, seed) — GET is
  reserved for safe reads (search).
- **multimodal is an optional extra** (torch is ~1–2 GB). The base app boots without it via
  lazy imports; the file-upload route is registered only when `python-multipart` is present.

## Environment / gotchas learned the hard way
- **Proxy**: the laptop set `HTTP(S)_PROXY=____`, which only resolves on
  VPN. It broke `uv`, `gh`, `git push`, and Hugging Face downloads whenever off-VPN. Fix:
  `unset` the proxy vars in that terminal, or use `make *-offvpn` (which also points uv at
  public PyPI). **The new laptop's network may differ — re-verify.**
- **git push HTTP 400**: chunked upload mangled by the proxy → `git config http.postBuffer
  524288000` fixed it.
- **PyCharm**: use a FastAPI run config with the **venv interpreter directly** (a *uv*
  interpreter routes through uv and hit the artifactory). No `--reload` when debugging.
- **Hugging Face Xet**: the new Xet transfer backend fails behind the proxy → set
  `HF_HUB_DISABLE_XET=1` (done in `clip_model.py`). Once CLIP is cached, run with
  `HF_HUB_OFFLINE=1` so it doesn't do flaky network update-checks.
- **ML model quirks**: `register?deploy=true` can report COMPLETED before the model is
  actually loaded for inference ("Model not ready yet") → `wait_until_ready` polls a tiny
  predict and nudges an explicit `_deploy`. `.plugins-ml-model` stores content **chunks**
  (no `model_state`) alongside the metadata doc; `find_deployed_model` must exclude chunks or
  it never detects an existing model and re-registers duplicates every run.
- **git identity**: commits use the GitHub privacy alias
  `22791991+anand4252@users.noreply.github.com` (set repo-local; global stays the office
  identity). Early commits were rewritten to strip the office email before the first push.

## Roadmap
1. **Conversational RAG** — retrieval (reuse a chapter) + an **LLM** to generate an answer
   from the retrieved text. Introduces **connectors** (OpenSearch → external/local model) and
   likely a local LLM (e.g. Ollama) to stay offline. This is where the deferred connector
   learning lands.
2. **Agentic search** — an LLM orchestrating tools/searches; builds on RAG.
3. **Building AI-search workflows in Dashboards** — a UI/tooling capstone over what exists.

## Resuming on a new laptop
1. `git clone` the repo (this doc + all code come with it).
2. Follow `README.md` setup: `uv sync` (`--extra multimodal` for CLIP), `make up`, `make
   bootstrap`, `make prepare`, `make seed`, etc.
3. Re-fetch the Flickr images (Kaggle) and copy the subset; models re-download on first use.
4. Start Claude Code in the repo — `CLAUDE.md` auto-loads the context; this journal fills in
   the "why."
