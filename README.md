# Voter ID / EPIC Assistant

Chatbot for Indian voter ID (EPIC) help using Retrieval-Augmented Generation (RAG) over official PDF documents.

## Features
- Parses official PDF files from the `data/` folder.
- Builds semantic search index using:
  - `sentence-transformers/all-MiniLM-L6-v2`
  - `chromadb`
- Retrieves relevant chunks for each user query.
- Generates grounded answers using either:
  - Google Gemini 1.5 Flash
  - Groq Llama-3 (free)
- Shows source citations for each response.
- Optional Hindi response mode in the UI.

## Project Structure
- `app/main.py`: Streamlit chat app.
- `src/voter_assistant/knowledge_assistant.py`: PDF ingestion, chunking, embeddings, indexing, retrieval, and answer generation.
- `src/project_config/assistant_settings.py`: Project-level default settings.
- `notebooks/voter_id_assistant_notebook.ipynb`: Notebook walkthrough of the same system.
- `data/`: Official PDF files used as source documents.
- `requirements.txt`: Project dependencies.

## Setup
1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run
```bash
streamlit run app/main.py
```

## Usage
1. Open the app.
2. Ask a voter-related question.
3. Review answer plus source citations.

## Admin Configuration (No User Key Entry)
Create a file named `.env` in project root.

You can copy from `.env.example` and fill real key values.

Required:
- `LLM_PROVIDER=google` or `LLM_PROVIDER=groq`
- For Google: `GOOGLE_API_KEY=...`
- For Groq: `GROQ_API_KEY=...`

Optional:
- `HINDI_MODE=true` or `false`
- `RETRIEVED_CHUNKS=4` (range 2 to 8)

Example `.env` content:

```env
LLM_PROVIDER=google
GOOGLE_API_KEY=your_key_here
GROQ_API_KEY=
HINDI_MODE=false
RETRIEVED_CHUNKS=4
```

Then run:

```bash
streamlit run app/main.py
```

For deployment:
- Streamlit Community Cloud: add these values in app Secrets.
- Hugging Face Spaces: add them in Space Settings > Variables/Secrets.

## Notes
- Source documents are taken only from local PDFs in `data/`.
- Use the "Rebuild PDF Index" button when PDF files are added or changed.

## Streamlit Cloud Secrets

Set secrets in Streamlit Community Cloud under **App settings -> Secrets**:

```toml
LLM_PROVIDER = "google"
GOOGLE_API_KEY = "your_google_api_key"
GROQ_API_KEY = ""
HINDI_MODE = "false"
RETRIEVED_CHUNKS = "4"
```

Do not commit real API keys to the repository. For local development, use `.env` or `.streamlit/secrets.toml`.
