import json
import os
import re
import sys
import csv
import datetime as dt
from pathlib import Path
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add project root to path
sys.path.append(os.path.join(BASE_DIR, ".."))

from src.voter_assistant.knowledge_assistant import RAGPipeline


def get_secret(name: str) -> Optional[str]:
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    value = os.getenv(name)
    return value if value else None


def run_tts(text: str, use_hindi_voice: bool, stop_only: bool = False) -> None:
    lang = "hi-IN" if use_hindi_voice else "en-IN"

    def _clean_for_speech(raw: str) -> str:
        cleaned = raw or ""
        # Remove inline source citations like [source:Form_6.pdf page:1]
        cleaned = re.sub(r"\[\s*source\s*:[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
        # Remove any trailing Sources section if model includes it.
        cleaned = re.split(r"\n\s*Sources\s*\n", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
        # Collapse repeated whitespace after cleanup.
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    spoken_text = _clean_for_speech(text)[:1800] if text else ""
    text_json = json.dumps(spoken_text)

    if stop_only:
        components.html(
            """
            <script>
                if (window.speechSynthesis) {
                    window.speechSynthesis.cancel();
                }
            </script>
            """,
            height=0,
        )
        return

    if not spoken_text:
        return

    components.html( f""" <script> (function() {{ if (!window.speechSynthesis) return; window.speechSynthesis.cancel(); const utter = new SpeechSynthesisUtterance({text_json}); utter.lang = '{lang}'; utter.rate = 1.0; utter.pitch = 1.0; window.speechSynthesis.speak(utter); }})(); </script> """, height=0, )


def build_retrieval_query(raw_query: str) -> str:
    query = (raw_query or "").strip()
    if not query:
        return ""

    if re.search(r"\bvoter\s*id\b", query, flags=re.IGNORECASE):
        return query

    return f"{query} voter id"


def _ensure_parent_dir(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)


def _append_metrics_row(file_path: Path, row: dict, header: list) -> None:
    _ensure_parent_dir(file_path)
    write_header = not file_path.exists()
    with file_path.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _bytes_to_mb(value: int) -> float:
    return round(value / (1024 * 1024), 2)


st.set_page_config(page_title="Voter ID / EPIC Assistant", page_icon="🗳️", layout="wide")
st.title("Voter ID / EPIC Assistant")
st.caption("Ask voter service questions grounded in official PDF documents.")

provider = (get_secret("LLM_PROVIDER") or "google").strip().lower()
default_hindi_mode = (get_secret("HINDI_MODE") or "false").strip().lower() == "true"
try:
    top_k = int(get_secret("RETRIEVED_CHUNKS") or "4")
except ValueError:
    top_k = 4
top_k = max(2, min(top_k, 8))

if provider not in {"google", "groq"}:
    provider = "google"

if provider == "google":
    api_key = (get_secret("GOOGLE_API_KEY") or "").strip()
else:
    api_key = (get_secret("GROQ_API_KEY") or "").strip()

with st.sidebar:
    st.header("System")
    st.caption(f"Provider: {provider}")
    hindi_mode = st.toggle("Talk in Hindi", value=default_hindi_mode)
    st.caption(f"Retrieved chunks: {top_k}")
    refresh_index = st.button("Rebuild PDF Index", disabled=st.session_state.get("processing", False))

data_dir = ROOT_DIR / "data"
db_dir = ROOT_DIR / "vector_store"
metrics_path = ROOT_DIR / "data" / "metrics.csv"
metrics_columns = [
    "timestamp",
    "provider",
    "query",
    "top_k",
    "hindi",
    "retrieve_ms",
    "generate_ms",
    "end_to_end_ms",
    "prompt_chars",
    "answer_chars",
    "prompt_tokens_est",
    "answer_tokens_est",
    "contexts",
    "rss_mb",
    "vms_mb",
]


@st.cache_resource(show_spinner=False)
def load_pipeline() -> RAGPipeline:
    return RAGPipeline(data_dir=data_dir, db_dir=db_dir)


pipeline = load_pipeline()

if refresh_index:
    with st.spinner("Rebuilding index from PDF files..."):
        index_metrics = pipeline.build_index(force_rebuild=True)
    st.success("Index rebuilt successfully.")
    if index_metrics:
        st.caption(
            "Index build: "
            f"total {index_metrics.get('index_total_ms', 0):.1f} ms, "
            f"extract {index_metrics.get('extract_ms', 0):.1f} ms, "
            f"embed {index_metrics.get('embed_ms', 0):.1f} ms, "
            f"upsert {index_metrics.get('upsert_ms', 0):.1f} ms, "
            f"chunks {index_metrics.get('chunks', 0)}."
        )

if "ready" not in st.session_state:
    with st.spinner("Preparing knowledge index..."):
        pipeline.build_index(force_rebuild=False)
    st.session_state.ready = True

if "messages" not in st.session_state:
    st.session_state.messages = []

if "tts_target_id" not in st.session_state:
    st.session_state.tts_target_id = None

if "tts_command" not in st.session_state:
    st.session_state.tts_command = None

if "processing" not in st.session_state:
    st.session_state.processing = False

if "index_metrics" not in st.session_state:
    st.session_state.index_metrics = None

for idx, msg in enumerate(st.session_state.messages, start=1):
    if "id" not in msg:
        msg["id"] = f"msg_{idx}"

cmd = st.session_state.tts_command
if cmd:
    action = cmd.get("action")
    if action == "stop":
        st.session_state.tts_target_id = None
        run_tts("", use_hindi_voice=hindi_mode, stop_only=True)
    elif action == "start":
        target_id = cmd.get("target_id")
        st.session_state.tts_target_id = target_id
        target_msg = next(
            (m for m in st.session_state.messages if m.get("id") == target_id and m["role"] == "assistant"),
            None,
        )
        if target_msg:
            run_tts(target_msg.get("content", ""), use_hindi_voice=hindi_mode)
    st.session_state.tts_command = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            is_active = st.session_state.tts_target_id == msg["id"]
            icon_name = ":material/volume_off:" if is_active else ":material/volume_up:"
            help_text = "Stop reading this message" if is_active else "Read this message aloud"
            if st.button(
                " ",
                key=f"tts_toggle_{msg['id']}",
                icon=icon_name,
                help=help_text,
                disabled=st.session_state.processing,
            ):
                if is_active:
                    st.session_state.tts_command = {"action": "stop"}
                else:
                    st.session_state.tts_command = {"action": "start", "target_id": msg["id"]}
                st.rerun()

query = st.chat_input("How do I apply for voter ID?")

if query:
    retrieval_query = build_retrieval_query(query)
    user_id = f"msg_{len(st.session_state.messages) + 1}"
    st.session_state.messages.append({"id": user_id, "role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if not api_key:
            answer_text = (
                "Service is not configured by admin yet. "
                "Set GOOGLE_API_KEY or GROQ_API_KEY in server environment."
            )
            st.markdown(answer_text)
            assistant_id = f"msg_{len(st.session_state.messages) + 1}"
            st.session_state.messages.append(
                {"id": assistant_id, "role": "assistant", "content": answer_text}
            )
            st.rerun()
        else:
            try:
                st.session_state.processing = True
                with st.spinner("Retrieving context and generating answer..."):
                    result = pipeline.answer(
                        query=retrieval_query,
                        provider=provider,
                        api_key=api_key,
                        top_k=top_k,
                        hindi=hindi_mode,
                    )
            except Exception as exc:
                error_text = f"Request failed: {exc}"
                st.error(error_text)
                assistant_id = f"msg_{len(st.session_state.messages) + 1}"
                st.session_state.messages.append(
                    {"id": assistant_id, "role": "assistant", "content": error_text}
                )
                st.rerun()
            finally:
                st.session_state.processing = False

            st.markdown(result["answer"])

            metrics = result.get("metrics", {})
            if metrics:
                try:
                    import psutil
                    process = psutil.Process(os.getpid())
                    mem_info = process.memory_info()
                    rss_mb = _bytes_to_mb(mem_info.rss)
                    vms_mb = _bytes_to_mb(mem_info.vms)
                except Exception:
                    rss_mb = 0.0
                    vms_mb = 0.0

                row = {
                    "timestamp": dt.datetime.utcnow().isoformat(timespec="seconds"),
                    "provider": provider,
                    "query": query,
                    "top_k": top_k,
                    "hindi": hindi_mode,
                    "retrieve_ms": round(metrics.get("retrieve_ms", 0.0), 2),
                    "generate_ms": round(metrics.get("generate_ms", 0.0), 2),
                    "end_to_end_ms": round(metrics.get("end_to_end_ms", 0.0), 2),
                    "prompt_chars": metrics.get("prompt_chars", 0),
                    "answer_chars": metrics.get("answer_chars", 0),
                    "prompt_tokens_est": metrics.get("prompt_tokens_est", 0),
                    "answer_tokens_est": metrics.get("answer_tokens_est", 0),
                    "contexts": metrics.get("contexts", 0),
                    "rss_mb": rss_mb,
                    "vms_mb": vms_mb,
                }
                _append_metrics_row(metrics_path, row, metrics_columns)

                with st.expander("Latency and size metrics"):
                    st.markdown(
                        "\n".join(
                            [
                                f"- Retrieve: {row['retrieve_ms']} ms",
                                f"- Generate: {row['generate_ms']} ms",
                                f"- End-to-end: {row['end_to_end_ms']} ms",
                                f"- Prompt chars: {row['prompt_chars']}",
                                f"- Answer chars: {row['answer_chars']}",
                                f"- Prompt tokens (est.): {row['prompt_tokens_est']}",
                                f"- Answer tokens (est.): {row['answer_tokens_est']}",
                                f"- Contexts: {row['contexts']}",
                                f"- RSS MB: {row['rss_mb']}",
                                f"- VMS MB: {row['vms_mb']}",
                            ]
                        )
                    )

            if result["citations"]:
                st.markdown("**Sources**")
                for item in result["citations"]:
                    st.markdown(f"- {item}")

            with st.expander("Retrieved context"):
                for idx, ctx in enumerate(result["contexts"], start=1):
                    st.markdown(f"**{idx}. {ctx['source']} (page {ctx['page']})**")
                    st.write(ctx["text"])

            assistant_id = f"msg_{len(st.session_state.messages) + 1}"
            st.session_state.messages.append(
                {"id": assistant_id, "role": "assistant", "content": result["answer"]}
            )
            st.rerun()
