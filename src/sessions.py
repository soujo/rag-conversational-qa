import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.chat_message_histories import ChatMessageHistory

from .config import HISTORY_DIR, INDEX_DIR, SESSIONS_FILE, UPLOADS_DIR, ensure_data_dirs
from .utils import (
    dicts_to_messages,
    ensure_dir,
    load_json,
    messages_to_dicts,
    save_json,
    utc_timestamp,
)


def load_sessions() -> List[Dict]:
    ensure_data_dirs()
    sessions = load_json(SESSIONS_FILE, [])
    return sessions if isinstance(sessions, list) else []


def save_sessions(sessions: List[Dict]) -> None:
    ensure_dir(SESSIONS_FILE.parent)
    save_json(SESSIONS_FILE, sessions)


def create_session(name: Optional[str] = None) -> Dict:
    sessions = load_sessions()
    now = utc_timestamp()
    session = {
        "id": str(uuid.uuid4()),
        "name": name or f"Session {len(sessions) + 1}",
        "created_at": now,
        "updated_at": now,
        "sources_hash": None,
        "sources": {"pdfs": [], "youtube": [], "web": []},
    }
    sessions.append(session)
    save_sessions(sessions)
    return session


def get_session(session_id: str) -> Optional[Dict]:
    for session in load_sessions():
        if session.get("id") == session_id:
            return session
    return None


def rename_session(session_id: str, new_name: str) -> None:
    sessions = load_sessions()
    for session in sessions:
        if session.get("id") == session_id:
            session["name"] = new_name or session["name"]
            session["updated_at"] = utc_timestamp()
    save_sessions(sessions)


def delete_session(session_id: str) -> None:
    sessions = [s for s in load_sessions() if s.get("id") != session_id]
    save_sessions(sessions)
    history_path = HISTORY_DIR / f"{session_id}.json"
    if history_path.exists():
        history_path.unlink()
    index_path = INDEX_DIR / session_id
    if index_path.exists():
        shutil.rmtree(index_path, ignore_errors=True)
    uploads_path = UPLOADS_DIR / session_id
    if uploads_path.exists():
        shutil.rmtree(uploads_path, ignore_errors=True)


def clear_session_history(session_id: str) -> None:
    history_path = HISTORY_DIR / f"{session_id}.json"
    if history_path.exists():
        history_path.unlink()


def clear_session_sources(session_id: str) -> None:
    index_path = INDEX_DIR / session_id
    uploads_path = UPLOADS_DIR / session_id
    if index_path.exists():
        shutil.rmtree(index_path, ignore_errors=True)
    if uploads_path.exists():
        shutil.rmtree(uploads_path, ignore_errors=True)
    update_session_sources(session_id, pdf_filenames=[], youtube_links=[], web_links=[], sources_hash=None)


def update_session_sources(
    session_id: str,
    pdf_filenames: List[str],
    youtube_links: List[str],
    web_links: List[str],
    sources_hash: Optional[str],
) -> None:
    sessions = load_sessions()
    for session in sessions:
        if session.get("id") == session_id:
            session["sources"] = {
                "pdfs": pdf_filenames,
                "youtube": youtube_links,
                "web": web_links,
            }
            session["sources_hash"] = sources_hash
            session["updated_at"] = utc_timestamp()
    save_sessions(sessions)


def load_chat_history(session_id: str) -> ChatMessageHistory:
    history_path = HISTORY_DIR / f"{session_id}.json"
    items = load_json(history_path, [])
    messages = dicts_to_messages(items if isinstance(items, list) else [])
    return ChatMessageHistory(messages=messages)


def save_chat_history(session_id: str, history: ChatMessageHistory) -> None:
    history_path = HISTORY_DIR / f"{session_id}.json"
    save_json(history_path, messages_to_dicts(history.messages))
