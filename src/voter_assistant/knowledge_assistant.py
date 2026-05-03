from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import re

import chromadb
import google.generativeai as genai
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    source: str
    page: int
    chunk_index: int


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(text):
        part = text[start : start + chunk_size].strip()
        if part:
            chunks.append(part)
        start += step
    return chunks


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


@st.cache_data(show_spinner=False)
def load_chunk_records(data_dir_str: str, chunk_size: int, chunk_overlap: int) -> List[ChunkRecord]:
    data_dir = Path(data_dir_str)
    pdf_files = sorted(data_dir.glob("*.pdf"))
    records: List[ChunkRecord] = []

    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))
        for page_idx, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = _clean_text(raw)
            if not cleaned:
                continue

            pieces = _chunk_text(
                text=cleaned,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            for c_idx, text in enumerate(pieces, start=1):
                cid = f"{pdf_path.name}_p{page_idx}_c{c_idx}"
                records.append(
                    ChunkRecord(
                        chunk_id=cid,
                        text=text,
                        source=pdf_path.name,
                        page=page_idx,
                        chunk_index=c_idx,
                    )
                )

    return records


@st.cache_resource(show_spinner=False)
def get_chroma_client(db_dir_str: str):
    return chromadb.PersistentClient(path=db_dir_str)


@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str):
    return SentenceTransformer(model_name)


class RAGPipeline:
    def __init__(
        self,
        data_dir: Path,
        db_dir: Path,
        collection_name: str = "epic_docs",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 850,
        chunk_overlap: int = 120,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.db_dir = Path(db_dir)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.db_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._embedder = None

    def _get_client(self):
        if self._client is None:
            self._client = get_chroma_client(str(self.db_dir))
        return self._client

    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder(self.embedding_model_name)
        return self._embedder

    def _extract_chunks(self) -> List[ChunkRecord]:
        return load_chunk_records(str(self.data_dir), self.chunk_size, self.chunk_overlap)

    def _get_or_create_collection(self):
        client = self._get_client()
        try:
            return client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            return client.get_or_create_collection(name=self.collection_name)

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embedder = self._get_embedder()
        embeddings = embedder.encode(texts, show_progress_bar=False)
        vectors = np.asarray(embeddings, dtype=np.float32)
        vectors = _l2_normalize(vectors)
        return vectors.tolist()

    def build_index(self, force_rebuild: bool = False) -> None:
        client = self._get_client()
        if force_rebuild:
            try:
                client.delete_collection(name=self.collection_name)
            except Exception:
                pass

        collection = self._get_or_create_collection()

        try:
            if not force_rebuild and collection.count() > 0:
                return
        except Exception:
            pass

        records = self._extract_chunks()
        if not records:
            return

        docs = [r.text for r in records]
        vectors = self._embed_texts(docs)

        collection.upsert(
            ids=[r.chunk_id for r in records],
            documents=docs,
            embeddings=vectors,
            metadatas=[
                {
                    "source": r.source,
                    "page": r.page,
                    "chunk_index": r.chunk_index,
                }
                for r in records
            ],
        )

        try:
            if hasattr(client, "persist"):
                client.persist()
        except Exception:
            pass

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict]:
        collection = self._get_or_create_collection()
        try:
            if collection.count() == 0:
                return []
        except Exception:
            pass

        query_embedding = self._embed_texts([query])[0]
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        contexts: List[Dict] = []
        for doc, meta, dist in zip(docs, metas, dists):
            contexts.append(
                {
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "page": int(meta.get("page", -1)),
                    "score": float(1 - dist) if dist is not None else 0.0,
                }
            )
        return contexts

    @staticmethod
    def _build_prompt(query: str, contexts: List[Dict], hindi: bool) -> str:
        evidence_lines = []
        for i, item in enumerate(contexts, start=1):
            evidence_lines.append(
                f"[{i}] Source: {item['source']}, Page: {item['page']}\n"
                f"Text: {item['text']}"
            )

        language_line = (
            (
                "Respond strictly in Hindi (Devanagari script). "
                "Do not switch to English except unavoidable form/code names."
            )
            if hindi
            else "Respond in clear English with simple wording."
        )

        return (
            "You are an assistant for Indian voter ID (EPIC) support.\n"
            "Answer only from the evidence given below.\n"
            "If evidence is insufficient, say that clearly and do not invent details.\n"
            "If the user asks for required documents, return a bullet list only from evidence text.\n"
            "For every factual point, add citation in this format: [source:FILE_NAME page:PAGE_NO].\n"
            f"{language_line}\n\n"
            f"User question:\n{query}\n\n"
            "Evidence:\n"
            + "\n\n".join(evidence_lines)
        )

    @staticmethod
    def _google_model_candidates() -> List[str]:
        # Keep only Gemini 1.5 Flash variants to match assignment constraints.
        return [
            "gemini-1.5-flash",
            "models/gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "models/gemini-1.5-flash-latest",
        ]

    @staticmethod
    def _generate_with_google(api_key: str, prompt: str) -> str:
        if not api_key:
            raise RuntimeError("Google API key not provided")
        genai.configure(api_key=api_key)

        tried: List[str] = []
        last_exc: Exception | None = None

        try:
            listed_models = []
            for model in genai.list_models():
                methods = getattr(model, "supported_generation_methods", []) or []
                if "generateContent" in methods:
                    listed_models.append(model.name)
            for name in listed_models:
                if name not in tried:
                    tried.append(name)
        except Exception:
            # If model listing fails, continue with fallback candidates.
            pass

        for candidate in RAGPipeline._google_model_candidates():
            if candidate not in tried:
                tried.append(candidate)

        for model_name in tried:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return (response.text or "").strip()
            except Exception as exc:
                last_exc = exc
                # Try next model name on any failure; this handles 404 model-not-found gracefully.
                continue

        raise RuntimeError(
            "Google model call failed for all available candidates. "
            f"Tried: {', '.join(tried)}. Last error: {last_exc}"
        )

    @staticmethod
    def _generate_with_groq(api_key: str, prompt: str) -> str:
        if not api_key:
            raise RuntimeError("Groq API key not provided")
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a grounded retrieval assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content.strip()

    def _generate(self, provider: str, api_key: str, prompt: str) -> str:
        if provider == "google":
            return self._generate_with_google(api_key=api_key, prompt=prompt)
        if provider == "groq":
            return self._generate_with_groq(api_key=api_key, prompt=prompt)
        return "Unsupported provider. Choose either google or groq."

    @staticmethod
    def _contains_devanagari(text: str) -> bool:
        return bool(re.search(r"[\u0900-\u097F]", text or ""))

    @staticmethod
    def _build_hindi_rewrite_prompt(answer: str) -> str:
        return (
            "Translate the following answer to Hindi (Devanagari script).\n"
            "Rules:\n"
            "1) Keep meaning exactly the same.\n"
            "2) Do not add new facts.\n"
            "3) Keep citation brackets exactly unchanged, for example [source:... page:...].\n"
            "4) Keep bullet points structure as-is.\n\n"
            "Answer to translate:\n"
            f"{answer}"
        )

    @staticmethod
    def _dedupe_citations(contexts: List[Dict]) -> List[str]:
        seen = set()
        unique: List[str] = []
        for item in contexts:
            label = f"{item['source']} (page {item['page']})"
            if label not in seen:
                seen.add(label)
                unique.append(label)
        return unique

    def answer(
        self,
        query: str,
        provider: str,
        api_key: str,
        top_k: int = 4,
        hindi: bool = False,
    ) -> Dict:
        contexts = self.retrieve(query=query, top_k=top_k)
        if not contexts:
            return {
                "answer": "No indexed PDF content is available yet. Please rebuild the index and try again.",
                "citations": [],
                "contexts": [],
            }

        prompt = self._build_prompt(query=query, contexts=contexts, hindi=hindi)

        answer = self._generate(provider=provider, api_key=api_key, prompt=prompt)

        # Some model responses ignore language instruction intermittently.
        # Enforce Hindi output when toggle is enabled without changing facts.
        if hindi and not self._contains_devanagari(answer):
            rewrite_prompt = self._build_hindi_rewrite_prompt(answer)
            rewritten = self._generate(provider=provider, api_key=api_key, prompt=rewrite_prompt)
            if rewritten.strip():
                answer = rewritten.strip()

        return {
            "answer": answer,
            "citations": self._dedupe_citations(contexts),
            "contexts": contexts,
        }
