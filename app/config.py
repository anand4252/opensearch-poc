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

    # Neural sparse QUERY tokenizer (doc-only): tokenizes the query at search time.
    # OpenSearch 2.19 does NOT support the `analyzer` field on neural_sparse queries;
    # doc-only search there references this lightweight tokenizer model by id instead.
    query_model_name: str = (
        "amazon/neural-sparse/opensearch-neural-sparse-tokenizer-v1"
    )
    query_model_version: str = "1.0.1"
    query_model_id: str = ""

    # Hybrid score combination weights: [lexical, sparse]
    lexical_weight: float = 0.3
    sparse_weight: float = 0.7

    # Field names used across the index/pipelines/queries
    text_field: str = "combined_text"
    embedding_field: str = "combined_text_embedding"


def get_settings() -> Settings:
    return Settings()
