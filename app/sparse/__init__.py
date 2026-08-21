"""Neural sparse search POC (BM25 + sparse_encoding, doc-only) with hybrid combination.

Doc-only mode: a sparse-encoding model expands documents into token->weight features
at ingest time; the query side uses a built-in analyzer (no query-time model inference).
Sits alongside the dense `app.semantic` POC on the same Flickr dataset.
"""
