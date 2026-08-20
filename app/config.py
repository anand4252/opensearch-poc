"""Application settings loaded from environment / .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Connection
    opensearch_url: str = "http://localhost:9200"

    # Index & pipeline names
    index_name: str = "hybrid-sparse-index"
    ingest_pipeline: str = "hybrid-sparse-ingest"
    search_pipeline: str = "hybrid-sparse-search"

    # Neural sparse INGEST model (doc-only): expands documents into token -> weight.
    model_name: str = (
        "amazon/neural-sparse/opensearch-neural-sparse-encoding-doc-v3-distill"
    )
    model_version: str = "1.0.0"
    model_format: str = "TORCH_SCRIPT"
    model_id: str = ""

    # Neural sparse QUERY side (doc-only): OpenSearch 3.x tokenizes the query with a
    # built-in analyzer (no query-time model inference), so no tokenizer model is needed.
    query_analyzer: str = "bert-uncased"

    # Hybrid score combination weights: [lexical, sparse]
    lexical_weight: float = 0.3
    sparse_weight: float = 0.7

    # Field names used across the index/pipelines/queries
    text_field: str = "combined_text"
    embedding_field: str = "combined_text_embedding"

    # Dataset / seeding. `seed_data_file` is what /documents/seed indexes; the Flickr
    # subset is built from `flickr_csv` by scripts/prepare_flickr.py (see README).
    seed_data_file: str = "data/flickr_docs.json"
    flickr_csv: str = "data/results.csv"
    flickr_subset_size: int = 1000
    flickr_images_dir: str = "data/images"


def get_settings() -> Settings:
    return Settings()
