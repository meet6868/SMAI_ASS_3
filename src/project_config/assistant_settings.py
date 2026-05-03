"""Project defaults used by the chatbot pipeline."""

from dataclasses import dataclass


@dataclass
class ProjectDefaults:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store: str = "chromadb"
    llm_provider: str = "google"
    llm_model: str = "gemini-1.5-flash"
    source_type: str = "official_pdf_documents"


def get_project_defaults() -> ProjectDefaults:
    return ProjectDefaults()
