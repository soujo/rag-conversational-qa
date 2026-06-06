import os
from pathlib import Path

import streamlit as st

from .utils import ensure_dir

DATA_DIR = Path("data")
SESSIONS_FILE = DATA_DIR / "sessions.json"
HISTORY_DIR = DATA_DIR / "history"
INDEX_DIR = DATA_DIR / "index"
UPLOADS_DIR = DATA_DIR / "uploads"

APP_TITLE = "Conversational RAG Chat"
APP_DESCRIPTION = "Chat normally or ground answers on your PDFs, YouTube links, and webpages."

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Default to a currently supported Groq model; adjust if you prefer another.
GROQ_MODEL = "llama-3.1-8b-instant"

TOP_K_DEFAULT = 6
TOP_K_MIN = 1
TOP_K_MAX = 10


def ensure_data_dirs() -> None:
    for path in [DATA_DIR, HISTORY_DIR, INDEX_DIR, UPLOADS_DIR]:
        ensure_dir(path)


def get_groq_api_key() -> str:
    key = None
    # Streamlit raises if no secrets file; guard with try/except
    try:
        if hasattr(st, "secrets"):
            key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        key = None
    key = key or os.getenv("GROQ_API_KEY")
    if not key:
        st.warning("Missing GROQ_API_KEY in Streamlit secrets or environment. Please set it to continue.")
        st.stop()
    return key
