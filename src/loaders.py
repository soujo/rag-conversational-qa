import re
from pathlib import Path
from typing import List, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from .config import UPLOADS_DIR
from .utils import ensure_dir, safe_filename

DEFAULT_YOUTUBE_LANGUAGES = ["en"]


class LoaderError(RuntimeError):
    pass


def parse_links(raw: str) -> List[str]:
    if not raw:
        return []
    parts = re.split(r"[\n,]+", raw)
    links: List[str] = []
    seen: set[str] = set()
    for part in parts:
        link = part.strip()
        if not link or link in seen:
            continue
        seen.add(link)
        links.append(link)
    return links


def _validate_http_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise LoaderError(f"Invalid URL: {url}")
    return url.strip()


def extract_youtube_id(url: str) -> str:
    parsed = urlparse(_validate_http_url(url))
    host = parsed.netloc.lower().replace("www.", "")

    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/", maxsplit=1)[0]
        if video_id:
            return video_id

    if host in {"youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
            if video_id:
                return video_id
        if parsed.path.startswith("/shorts/") or parsed.path.startswith("/live/"):
            video_id = parsed.path.strip("/").split("/", maxsplit=1)[1]
            if video_id:
                return video_id

    raise LoaderError(f"Could not parse YouTube video id from URL: {url}")


def _load_pdf_documents(path: Path, source_name: str) -> List[Document]:
    try:
        pdf_docs = PyPDFLoader(str(path)).load()
    except Exception as exc:  # noqa: BLE001
        raise LoaderError(f"Failed to read PDF '{source_name}'.") from exc

    for doc in pdf_docs:
        doc.metadata.update({"type": "PDF", "source": source_name, "file_path": str(path)})
    return pdf_docs


def _transcript_items_to_text(items) -> str:
    snippets = []
    for item in items:
        if isinstance(item, dict):
            text = item.get("text", "")
        else:
            text = getattr(item, "text", "")
        text = text.strip()
        if text:
            snippets.append(text)
    return " ".join(snippets)


def _fetch_transcript_items(video_id: str):
    api = YouTubeTranscriptApi()

    if hasattr(api, "fetch"):
        fetched = api.fetch(video_id, languages=DEFAULT_YOUTUBE_LANGUAGES)
        if hasattr(fetched, "to_raw_data"):
            return fetched.to_raw_data()
        return list(fetched)

    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        return YouTubeTranscriptApi.get_transcript(video_id, languages=DEFAULT_YOUTUBE_LANGUAGES)

    raise LoaderError("Installed youtube-transcript-api version is not supported.")


def load_pdfs(session_id: str, uploaded_files) -> Tuple[List[Document], List[Path]]:
    docs: List[Document] = []
    saved_paths: List[Path] = []
    dest_dir = UPLOADS_DIR / session_id
    ensure_dir(dest_dir)
    for file in uploaded_files:
        filename = safe_filename(file.name)
        dest = dest_dir / filename
        try:
            with dest.open("wb") as f:
                f.write(file.getbuffer())
            pdf_docs = _load_pdf_documents(dest, filename)
        except Exception as exc:  # noqa: BLE001
            raise LoaderError(f"Failed to ingest uploaded PDF '{file.name}'.") from exc
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
        pdf_docs = _load_pdf_documents(path, name)
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
            transcript_items = _fetch_transcript_items(video_id)
        except (TranscriptsDisabled, NoTranscriptFound):
            raise LoaderError(f"Transcript not available for {url}")
        except Exception as exc:  # noqa: BLE001
            raise LoaderError(f"Failed to fetch transcript for {url}: {exc}") from exc
        text = _transcript_items_to_text(transcript_items)
        if not text:
            raise LoaderError(f"Transcript was empty for {url}")
        doc = Document(
            page_content=text,
            metadata={"type": "YouTube", "source": url, "video_id": video_id},
        )
        documents.append(doc)
    return documents


def load_webpages(urls: Sequence[str]) -> List[Document]:
    clean_urls = [_validate_http_url(u) for u in urls if u]
    if not clean_urls:
        return []
    documents: List[Document] = []
    for url in clean_urls:
        try:
            docs = WebBaseLoader(url).load()
        except Exception as exc:  # noqa: BLE001
            raise LoaderError(f"Failed to load webpage {url}: {exc}") from exc
        for doc in docs:
            doc.metadata.update({"type": "Web", "source": doc.metadata.get("source") or url})
        documents.extend(docs)
    return documents
