import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def utc_timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return cleaned or "file"


def compute_sources_hash(
    pdf_paths: Iterable[Path], youtube_links: Sequence[str], web_links: Sequence[str]
) -> str:
    parts: List[str] = []
    for path in sorted({Path(p) for p in pdf_paths}):
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            size = 0
        parts.append(f"pdf:{path.name}:{size}")
    for link in sorted({l.strip() for l in youtube_links if l.strip()}):
        parts.append(f"yt:{link}")
    for link in sorted({l.strip() for l in web_links if l.strip()}):
        parts.append(f"web:{link}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def messages_to_dicts(messages):
    data = []
    for msg in messages:
        role = "user"
        if isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, SystemMessage):
            role = "system"
        content = msg.content
        timestamp = getattr(msg, "timestamp", None) or utc_timestamp()
        data.append({"role": role, "content": content, "timestamp": timestamp})
    return data


def dicts_to_messages(items):
    messages = []
    for item in items:
        role = item.get("role")
        content = item.get("content", "")
        if role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "system":
            messages.append(SystemMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return default


def save_json(path: Path, payload) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
