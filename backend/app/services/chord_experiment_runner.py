from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run_experiment_lab(audio: Path, chord_dict: str, igtgs_root: str | None) -> tuple[str, dict[str, Any]]:
    """
    以子程序執行專案根目錄的 experiment.py，產生 JSON + .lab。
    """
    repo = _repo_root()
    script = repo / "experiment.py"
    if not script.is_file():
        raise FileNotFoundError(f"找不到 {script}")

    audio = audio.resolve()
    if not audio.is_file():
        raise FileNotFoundError(f"找不到音訊：{audio}")

    with tempfile.TemporaryDirectory(prefix="chord_exp_") as td_raw:
        td = Path(td_raw)
        json_path = td / "out.json"
        lab_path = td / "out.lab"

        env = os.environ.copy()
        py_paths = [str(repo)]
        if igtgs_root and igtgs_root.strip():
            igr = str(Path(igtgs_root).resolve())
            py_paths.insert(0, igr)
            env["IGTGS_ROOT"] = igr
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(py_paths + ([prev] if prev else []))

        cmd = [
            sys.executable,
            str(script),
            str(audio),
            "-o",
            str(json_path),
            "--lab-out",
            str(lab_path),
            "--chord-dict",
            chord_dict,
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
            raise RuntimeError(f"experiment.py 失敗：{err}")

        if not lab_path.is_file():
            raise RuntimeError("experiment 未產生 .lab")
        lab_text = lab_path.read_text(encoding="utf-8")
        if not json_path.is_file():
            raise RuntimeError("experiment 未產生 JSON")
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        return lab_text, meta
