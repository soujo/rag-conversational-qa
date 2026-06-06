import re
from pathlib import Path
from typing import List, Sequence, Tuple

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from .config import UPLOADS_DIR
from .utils import ensure_dir, safe_filename


def parse_links(raw: str) -> List[str]:
    if not raw:
        return []
    parts = re.split(r"[\n,]+", raw)
    return [p.strip() for p in parts if p.strip()]


def extract_youtube_id(url: str) -> str:
    patterns = [
        r"youtu\.be/([A-Za-z0-9_-]{6,})",
        r"youtube\.com/watch\?v=([A-Za-z0-9_-]{6,})",
        r"youtube\.com/shorts/([A-Za-z0-9_-]{6,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not parse YouTube video id from URL: {url}")


def load_pdfs(session_id: str, uploaded_files) -> Tuple[List[Document], List[Path]]:
    docs: List[Document] = []
    saved_paths: List[Path] = []
    dest_dir = UPLOADS_DIR / session_id
    ensure_dir(dest_dir)
    for file in uploaded_files:
        filename = safe_filename(file.name)
        dest = dest_dir / filename
        with dest.open("wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(str(dest))
        pdf_docs = loader.load()
        for doc in pdf_docs:
            doc.metadata.update(
                {"type": "PDF", "source": filename, "file_path": str(dest)}
            )
        docs.extend(pdf_docs)
        saved_paths.append(dest)
    return docs, saved_paths


def load_existing_pdfs(session_id: str, filenames: Sequence[str]) -> Tuple[List[Document], List[Path]]:
    docs: List[Document] = []
    paths: List[Path] = []
    for name in filenames:
        path = UPLOADS_DIR / session_id / name
        if not path.exists():
            continue
        loader = PyPDFLoader(str(path))
        pdf_docs = loader.load()
        for doc in pdf_docs:
            doc.metadata.update(
                {"type": "PDF", "source": name, "file_path": str(path)}
            )
        docs.extend(pdf_docs)
        paths.append(path)
    return docs, paths


def load_youtube_transcripts(urls: Sequence[str]) -> List[Document]:
    documents: List[Document] = []
    for url in urls:
        if not url:
            continue
        video_id = extract_youtube_id(url)
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        except (TranscriptsDisabled, NoTranscriptFound):
            raise ValueError(f"Transcript not available for {url}")
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Failed to fetch transcript for {url}: {exc}") from exc
        text = " ".join([item.get("text", "") for item in transcript_list])
        doc = Document(
            page_content=text,
            metadata={"type": "YouTube", "source": url, "video_id": video_id},
        )
        documents.append(doc)
    return documents


def load_webpages(urls: Sequence[str]) -> List[Document]:
    clean_urls = [u for u in urls if u]
    if not clean_urls:
        return []
    loader = WebBaseLoader(clean_urls)
    docs = loader.load()
    for doc in docs:
        doc.metadata.update({"type": "Web", "source": doc.metadata.get("source")})
    return docs
