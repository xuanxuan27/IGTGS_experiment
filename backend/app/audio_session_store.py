"""
YouTube 分析後的 session 音訊：記錄路徑與過期時間，並清掉逾期檔案。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

_lock = threading.Lock()
# session_id -> (wav path, expires_at unix time)
_sessions: dict[str, tuple[Path, float]] = {}


def _collect_expired(now: float) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for sid, (p, exp) in list(_sessions.items()):
        if exp <= now:
            out.append((sid, p))
    return out


def _unlink_safely(p: Path) -> None:
    try:
        if p.is_file():
            p.unlink()
    except OSError:
        pass


def cleanup_expired() -> int:
    """
    移除已過期的 session 記錄並刪除對應 WAV。回傳刪除的 session 數量。
    """
    now = time.time()
    with _lock:
        expired = _collect_expired(now)
        for sid, _ in expired:
            _sessions.pop(sid, None)
        removed = len(expired)
    for _, p in expired:
        _unlink_safely(p)
    return removed


def _cleanup_expired_locked() -> None:
    """已持有 _lock。"""
    now = time.time()
    expired = _collect_expired(now)
    for sid, _ in expired:
        _sessions.pop(sid, None)
    for _, p in expired:
        _unlink_safely(p)


def register_session(session_id: str, wav_path: Path, ttl_seconds: int) -> None:
    """登記可供 /api/audio/session 讀取的音訊檔與過期時間。"""
    ttl = max(60, int(ttl_seconds))
    exp = time.time() + ttl
    p = wav_path.resolve()
    with _lock:
        _cleanup_expired_locked()
        _sessions[session_id] = (p, exp)


def get_session_path(session_id: str) -> Path | None:
    """取得仍有效的音訊路徑；若過期或檔案不在則清記錄並回傳 None。"""
    cleanup_expired()
    stale: Path | None = None
    with _lock:
        t = _sessions.get(session_id)
        if not t:
            return None
        p, exp = t
        now = time.time()
        if now > exp:
            _sessions.pop(session_id, None)
            stale = p
        elif not p.is_file():
            _sessions.pop(session_id, None)
            return None
        else:
            return p
    if stale is not None:
        _unlink_safely(stale)
    return None
