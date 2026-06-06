from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import EMBEDDING_MODEL, INDEX_DIR
from .utils import ensure_dir

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100


class IndexingError(RuntimeError):
    pass


def split_documents(documents: List[Document]) -> List[Document]:
    clean_documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
    if not clean_documents:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(clean_documents)


@lru_cache(maxsize=1)
def get_embeddings():
    try:
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception as exc:  # noqa: BLE001
        raise IndexingError(
            f"Failed to initialize embedding model '{EMBEDDING_MODEL}'."
        ) from exc


def _index_files_exist(persist_dir: Path) -> bool:
    return (persist_dir / "index.faiss").exists() and (persist_dir / "index.pkl").exists()


def build_faiss_index(session_id: str, documents: List[Document]) -> Optional[FAISS]:
    if not documents:
        return None
    chunks = split_documents(documents)
    if not chunks:
        raise IndexingError("No valid text content was found in the selected sources.")
    embeddings = get_embeddings()
    persist_dir = INDEX_DIR / session_id
    ensure_dir(persist_dir)
    try:
        store = FAISS.from_documents(chunks, embeddings)
        store.save_local(str(persist_dir))
    except Exception as exc:  # noqa: BLE001
        raise IndexingError("Failed to build or save the FAISS index.") from exc
    return store


def load_faiss_index(session_id: str) -> Optional[FAISS]:
    persist_dir = INDEX_DIR / session_id
    if not persist_dir.exists() or not _index_files_exist(persist_dir):
        return None
    try:
        embeddings = get_embeddings()
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
