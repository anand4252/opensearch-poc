# Multimodal Search POC (CLIP, text → image)

The chapter where we finally search the **images themselves**, not their captions. We run
**CLIP** locally in our app to turn each Flickr image into a vector, store those vectors in
OpenSearch, and at search time turn the query *text* into a vector with the same model —
then ask OpenSearch for the nearest image vectors.

Code lives in `app/multimodal/`. Same 1,000-image subset as every other chapter.

---

## The big idea: one shared space for images and text

**CLIP** (Contrastive Language-Image Pre-training) was trained on ~400M image/caption
pairs so that an image and a matching description land **close together** in the same
vector space. That's what makes text → image search work: embed "a dog on the beach" as
text, embed the photos as images, and the nearest neighbors are the dogs on beaches — even
if nobody wrote that caption.

## How it differs from the earlier chapters

| | Sparse / Dense | **Multimodal** |
| --- | --- | --- |
| What we embed | the **captions** (text) | the **image pixels** |
| Model | text model, **inside** OpenSearch | **CLIP**, **in our app** |
| Who runs the model | OpenSearch (pipeline + `neural`) | **our Python code** |
| Ingest pipeline | yes | **none** — we index finished vectors |
| Query | `neural` (OpenSearch embeds) | plain **`knn`** (we pass the vector) |
| Captions | embedded & searched | kept only as **display text** |

So here OpenSearch is purely a **vector store**: it does the "find nearest" (kNN) math, and
all the *understanding* (pixels/words → vectors) happens in our app via CLIP.

## Why this design (Option B)

We deliberately run CLIP in the app rather than behind an OpenSearch connector, because
OpenSearch's built-in multimodal needs an **external cloud model** (Bedrock Titan / Cohere),
which breaks the fully-local, rebuild-from-clone goal. Running CLIP ourselves keeps
everything on the laptop and puts the model front-and-center — which is the point.

## The model

**`sentence-transformers/clip-ViT-B-32`** — 512-dim, cosine similarity. One `.encode()`
call handles both images and text. Heavy (pulls in `torch`), so it's an **opt-in extra**:
```
make sync-multimodal          # on VPN
make sync-multimodal-offvpn   # off VPN (public PyPI)
```
The base app boots without it — CLIP is imported lazily, only when a `/multimodal/*`
endpoint runs.

## Configuration (`app/config.py` / `.env`)

```
MULTIMODAL_INDEX_NAME=multimodal-clip-index
CLIP_MODEL_NAME=clip-ViT-B-32
CLIP_DIMENSION=512
CLIP_SPACE_TYPE=cosinesimil
CLIP_ENGINE=lucene
IMAGE_VECTOR_FIELD=image_vector
```

## The kNN index (that's the whole "bootstrap")

No model to deploy, no ingest pipeline — just an index shaped to hold CLIP vectors:
```json
"image_vector": {
  "type": "knn_vector",
  "dimension": 512,
  "method": { "name": "hnsw", "space_type": "cosinesimil", "engine": "lucene" }
}
```

## The query

```json
GET multimodal-clip-index/_search
{
  "size": 5,
  "_source": { "excludes": ["image_vector"] },
  "query": { "knn": { "image_vector": { "vector": [ ...512 floats... ], "k": 5 } } }
}
```
The 512 floats are the CLIP embedding of the query text, computed in our app.

---

## Running it

```bash
make up                     # OpenSearch (if not already up)
make sync-multimodal-offvpn # install CLIP/torch (one-time, ~1-2 GB)
make prepare                # build data/flickr_docs.json (shared; already done)
make copy-images IMAGES_SRC=~/Downloads/flickr30k_images/flickr30k_images   # copy the 1000 pixels
make bootstrap-multimodal   # create the kNN image index
make run                    # start the API
make seed-multimodal        # embed + index the 1000 images (slow first time: CLIP loads)
```

Then:
```bash
curl "localhost:8000/multimodal/search?q=a%20dog%20on%20the%20beach"
```

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/multimodal/copy-images` | copy the subset .jpgs from a source folder into `data/images` (`{images_src}`) |
| POST | `/multimodal/bootstrap` | create the kNN index (`?recreate_index=`) |
| POST | `/multimodal/seed` | embed the subset images with CLIP + index the vectors |
| POST/GET | `/multimodal/search` | text → image kNN search; `?q=…&size=…` |

**Demo entirely from Swagger:** `POST /dataset/prepare` → `POST /multimodal/copy-images`
(paste your Flickr images path) → `POST /multimodal/bootstrap` → `POST /multimodal/seed`
→ `GET /multimodal/search`. Same click-through shape as the sparse/dense chapters, plus
the image-copy step this chapter needs.

### The images (this chapter needs the pixels)
`make copy-images` copies the deterministic 1,000-image subset into `data/images/`
(gitignored). On a fresh laptop, re-fetch the Flickr images (Kaggle) or Git-LFS the
~250 MB subset, then re-run it.

## The fun comparison

Run the **same** query three ways and see how differently they retrieve:
- `/sparse/search?q=…&mode=sparse` — matches caption *words*
- `/semantic/search?q=…` — matches caption *meaning*
- `/multimodal/search?q=…` — matches the *image content itself*

Multimodal can surface a photo whose caption never used your words, because it "sees" the
picture.

## Troubleshooting the CLIP download

On first seed the app fetches CLIP's weights from Hugging Face. Two gotchas on locked-down
networks:

- **Xet backend fails behind a proxy.** HF's newer "Xet" transfer errors out (`xet_get` /
  `nodename nor servname`), even though plain HTTPS works. We force the classic path via
  `HF_HUB_DISABLE_XET=1` (set automatically in `app/multimodal/clip_model.py`), so the
  first download works over standard HTTPS.
- **Run offline once cached.** After the model is downloaded once (cached in
  `~/.cache/huggingface`), run the app with `HF_HUB_OFFLINE=1` so seed loads from the cache
  and makes **no** network calls — handy when the proxy/VPN state is flaky:
  ```bash
  export HF_HUB_OFFLINE=1
  make run
  ```

## Out of scope (next step)
**Image → image** search (upload a photo, find visually similar ones) — an easy follow-on,
since CLIP already embeds images. We'll add it after text → image is solid.
