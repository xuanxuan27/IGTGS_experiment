"""
解析 IGTGS 後端目錄（含 beat_service、chord_service、compat）。

chord_ui 根目錄可只放 analysis_engine、experiment、grid_builder；
實際引擎程式在並排 clone 的 IGTGS/igtgs_backend。
"""
from __future__ import annotations

import os
from pathlib import Path


def resolve_igtgs_backend_dir(engine_root: Path) -> Path:
    """
    engine_root：放 analysis_engine.py 的目錄（通常為 chord_ui 專案根）。

    優先序：
    1. 環境變數 IGTGS_BACKEND_DIR
    2. <engine_root>/igtgs_backend（若含 compat 子目錄）
    3. 與 chord_ui 同層 …/igtgs/igtgs_backend、…/IGTGS/igtgs_backend
    4. chord_ui 子目錄 igtgs/igtgs_backend
    """
    env = (os.environ.get("IGTGS_BACKEND_DIR") or "").strip()
    if env:
        return Path(env).expanduser().resolve()

    root = engine_root.resolve()
    local = (root / "igtgs_backend").resolve()
    if (local / "compat").is_dir():
        return local

    for cand in (
        root.parent / "igtgs",
        root.parent / "IGTGS",
        root / "igtgs",
    ):
        sub = (cand / "igtgs_backend").resolve()
        if sub.is_dir() and (sub / "compat").is_dir():
            return sub

    return local
