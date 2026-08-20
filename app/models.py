"""Pydantic request/response schemas for the API."""

from enum import StrEnum

from pydantic import BaseModel, Field


class SearchMode(StrEnum):
    lexical = "lexical"  # BM25 match only
    sparse = "sparse"    # neural_sparse only
    hybrid = "hybrid"    # both, combined via normalization pipeline


class Document(BaseModel):
    name: str
    combined_text: str


class IngestRequest(BaseModel):
    documents: list[Document]


class IngestResponse(BaseModel):
    indexed: int
    errors: int
    total_in_index: int


class PrepareResponse(BaseModel):
    images_available: int
    docs_written: int
    output_file: str


class BootstrapResponse(BaseModel):
    model_id: str
    model_reused: bool
    index_created: bool
    ingest_pipeline: str
    search_pipeline: str
    persisted: bool


class SearchRequest(BaseModel):
    query: str
    mode: SearchMode = SearchMode.hybrid
    size: int = Field(default=5, ge=1, le=50)


class SearchHit(BaseModel):
    score: float
    name: str
    combined_text: str


class SearchResponse(BaseModel):
    mode: SearchMode
    query: str
    hits: list[SearchHit]
