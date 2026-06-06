from pathlib import Path
from typing import List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .config import EMBEDDING_MODEL, INDEX_DIR
from .utils import ensure_dir

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_faiss_index(session_id: str, documents: List[Document]) -> Optional[FAISS]:
    if not documents:
        return None
    chunks = split_documents(documents)
    embeddings = get_embeddings()
    store = FAISS.from_documents(chunks, embeddings)
    persist_dir = INDEX_DIR / session_id
    ensure_dir(persist_dir)
    store.save_local(str(persist_dir))
    return store


def load_faiss_index(session_id: str) -> Optional[FAISS]:
    persist_dir = INDEX_DIR / session_id
    if not persist_dir.exists():
        return None
    embeddings = get_embeddings()
    try:
        return FAISS.load_local(
            str(persist_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception:  # noqa: BLE001
        return None


def get_retriever(store: Optional[FAISS], k: int):
    if store is None:
        return None
    return store.as_retriever(search_kwargs={"k": k})
