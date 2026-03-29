from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def require_ytdlp() -> str:
    path = shutil.which("yt-dlp")
    if not path:
        raise RuntimeError("找不到 yt-dlp，請安裝：pip install yt-dlp（並確認 ffmpeg 已安裝）")
    return path


def download_youtube_audio(url: str, work_dir: Path) -> Path:
    """
    下載為 WAV（需 ffmpeg）。回傳實際音訊檔路徑。
    """
    require_ytdlp()
    work_dir.mkdir(parents=True, exist_ok=True)
    template = str(work_dir / "yt_audio.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f",
        "bestaudio/best",
        "--no-playlist",
        "-x",
        "--audio-format",
        "wav",
        "-o",
        template,
        "--quiet",
        "--no-warnings",
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"yt-dlp 失敗：{err}")
    candidates = sorted(work_dir.glob("yt_audio.*"))
    for p in candidates:
        if p.is_file() and p.suffix.lower() in {".wav", ".m4a", ".opus", ".webm", ".mp3"}:
            return p
    raise RuntimeError("下載完成但找不到音訊檔")
