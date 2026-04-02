"""
應用設定：讀取環境變數，並在專案根目錄有 .env 時載入（不覆寫已存在的環境變數）。
無需安裝 pydantic-settings。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .igtgs_discovery import discover_igtgs_root


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    env_path = _repo_root() / ".env"
    if not env_path.is_file():
        return
    try:
        raw = env_path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    igtgs_root: str
    audio_session_ttl_seconds: int
    audio_sessions_dir: str


@lru_cache
def get_settings() -> Settings:
    ttl_raw = os.environ.get("AUDIO_SESSION_TTL_SECONDS", "3600")
    try:
        ttl = max(60, int(ttl_raw))
    except ValueError:
        ttl = 3600
    igtgs = (os.environ.get("IGTGS_ROOT") or "").strip()
    if not igtgs:
        igtgs = discover_igtgs_root(_repo_root())
    return Settings(
        igtgs_root=igtgs,
        audio_session_ttl_seconds=ttl,
        audio_sessions_dir=(os.environ.get("AUDIO_SESSIONS_DIR") or "").strip(),
    )
