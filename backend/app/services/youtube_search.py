from __future__ import annotations

import json
import subprocess
from urllib.parse import parse_qs, urlparse

from .youtube_audio import require_ytdlp


def _norm_video_id(obj: dict) -> str | None:
    """從 yt-dlp flat-playlist 條目取得 11 碼 video id。"""
    vid = (obj.get("id") or "").strip()
    if not vid:
        return None
    if len(vid) == 11 and all(c.isalnum() or c in "_-" for c in vid):
        return vid
    u = obj.get("url") or obj.get("webpage_url") or vid
    if "youtu.be/" in u:
        try:
            return u.split("youtu.be/")[1].split("?")[0].split("/")[0][:11]
        except IndexError:
            return None
    if "watch" in u:
        qs = parse_qs(urlparse(u).query)
        v = qs.get("v", [None])[0]
        if v and len(v) >= 11:
            return v[:11]
    return vid[:11] if len(vid) >= 11 else None


def youtube_search_videos(query: str, max_results: int = 10) -> list[dict]:
    """
    以 yt-dlp 搜尋 YouTube（ytsearchN:），不需官方 API 金鑰。
    每列一筆 JSON，欄位與舊 API 版盡量一致供前端沿用。
    """
    require_ytdlp()
    q = (query or "").strip()
    if not q:
        raise ValueError("搜尋關鍵字不可為空")
    n = max(1, min(int(max_results), 25))
    url = f"ytsearch{n}:{q}"
    cmd = [
        "yt-dlp",
        url,
        "--flat-playlist",
        "--dump-json",
        "--no-download",
        "--quiet",
        "--no-warnings",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=90,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"yt-dlp 搜尋失敗：{err}")

    items: list[dict] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("_type") == "playlist":
            for e in obj.get("entries") or []:
                if not isinstance(e, dict):
                    continue
                vid = _norm_video_id(e)
                if not vid:
                    continue
                items.append(_entry_to_item(e, vid))
            break
        vid = _norm_video_id(obj)
        if not vid:
            continue
        items.append(_entry_to_item(obj, vid))

    return items[:n]


def _entry_to_item(obj: dict, video_id: str) -> dict:
    title = obj.get("title") or ""
    channel = (
        obj.get("channel")
        or obj.get("uploader")
        or obj.get("channel_id")
        or ""
    )
    desc = (obj.get("description") or "")[:280]
    thumb = None
    thumbs = obj.get("thumbnails") or []
    if thumbs:
        thumb = thumbs[-1].get("url") or thumbs[0].get("url")
    return {
        "video_id": video_id,
        "title": title,
        "channel_title": str(channel),
        "description": desc,
        "thumbnail_default": thumb,
    }
