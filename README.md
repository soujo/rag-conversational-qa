# Conversational RAG Chat

Streamlit app for chatting with a Groq LLM, with optional RAG over PDFs, YouTube transcripts, and webpages.

## What It Does

- Normal chat when no sources are indexed
- RAG chat when sources are added and indexed
- Separate sessions with their own chat history and knowledge base
- Source support for PDFs, YouTube links, and web links
- Strict mode to avoid unsupported answers
- FAISS-based local vector search

## Tech Stack

- Streamlit
- LangChain
- Groq (`llama-3.1-8b-instant`)
- FAISS
- Hugging Face embeddings (`all-MiniLM-L6-v2`)

## Project Structure

```text
app.py               Main Streamlit app
requirements.txt     Python dependencies
src/config.py        App settings and paths
src/ui.py            Streamlit UI helpers
src/loaders.py       PDF, YouTube, and webpage loaders
src/indexing.py      Chunking, embeddings, FAISS index
src/prompts.py       Prompt templates
src/llm.py           Groq model setup
src/rag.py           Chat and RAG chains
src/sessions.py      Session and history persistence
src/utils.py         Shared helpers
```

## How It Works

1. Create or select a session.
2. Add PDFs, YouTube links, or webpage links.
3. Build the session index.
4. Ask questions.
5. If an index exists, answers use retrieved context. Otherwise, it behaves like normal chat.

## Data Stored Locally

The app creates a `data/` folder for:

- session metadata
- chat history
- uploaded PDFs
- FAISS indexes

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your_key"
streamlit run app.py
```
<!-- 
## Notes

- RAG is session-specific.
- YouTube indexing works only when transcripts are available.
- Web indexing depends on page accessibility.
- No tests are included yet. -->
