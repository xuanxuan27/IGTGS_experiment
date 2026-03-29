"""偵測本機 IGTGS 根目錄（含 analysis_engine）。"""
from __future__ import annotations

from pathlib import Path


def looks_like_igtgs_root(path: Path) -> bool:
    p = path.resolve()
    if (p / "analysis_engine.py").is_file():
        return True
    if (p / "analysis_engine" / "__init__.py").is_file():
        return True
    return False


def discover_igtgs_root(repo_root: Path) -> str:
    """
    若環境變數未設定，依序嘗試：
    <repo 的上一層>/igtgs、/IGTGS，以及 <repo>/igtgs。
    """
    root = repo_root.resolve()
    for cand in (
        root.parent / "igtgs",
        root.parent / "IGTGS",
        root / "igtgs",
    ):
        if looks_like_igtgs_root(cand):
            return str(cand.resolve())
    return ""
