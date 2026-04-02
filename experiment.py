#!/usr/bin/env python3
"""
實驗腳本：對單一音檔跑 IGTGS 內建和弦辨識（lvcr + madmom 節拍），
再依「時間軸上每一個和弦 segment」檢查 quality 是否為 maj / maj7 / min / min7；
若是則截取該段音訊送 ChordRefiner，若 softmax 最大值（信心）>= 0.5 則以 argmax 類別更新 quality（根音不變）。

本 repo 的專案根目錄即資料夾 chord_ui（與本檔同層為 backend/、frontend/、experiment.py 等）。
建議在此目錄執行：`python experiment.py ...`；`.env` 亦置於 chord_ui 根目錄。

環境變數：
  IGTGS_ROOT  含 analysis_engine 的 IGTGS 專案根（選用；未設則使用本目錄的精簡腳本）。

使用方式：
  python experiment.py /path/to/audio.wav
  python experiment.py /path/to/audio.m4a -o result.json --lab-out result.lab
  python experiment.py audio.wav --chord-dict custom --lab-out out.lab
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent


def _discover_igtgs_near_chord_ui() -> str:
    """與 backend/app/igtgs_discovery 相同規則，獨立寫在此以便單獨執行本腳本。"""
    root = BASE_DIR.resolve()
    for cand in (root.parent / "igtgs", root.parent / "IGTGS", root / "igtgs"):
        p = cand.resolve()
        if (p / "analysis_engine.py").is_file():
            return str(p)
        if (p / "analysis_engine" / "__init__.py").is_file():
            return str(p)
    return ""


IGTGS_ROOT = (os.environ.get("IGTGS_ROOT") or "").strip() or _discover_igtgs_near_chord_ui()

# IGTGS 目錄優先（含 analysis_engine），其次本腳本目錄
_module_paths: list[str] = []
if IGTGS_ROOT:
    _module_paths.append(str(Path(IGTGS_ROOT).resolve()))
_module_paths.append(str(BASE_DIR))
for _p in reversed(_module_paths):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from madmom_compat import prepare_for_madmom_import  # noqa: E402

# madmom：Python 3.10 collections、NumPy 1.24+ 別名；須先於間接 import madmom
prepare_for_madmom_import()

try:
    from analysis_engine import analyze_audio_file  # noqa: E402
    from beat_chord_refinement import (  # noqa: E402
        REFINER_CONFIDENCE_MIN,
        REFINE_QUALITIES,
        get_refiner_model,
        parse_root_quality,
        refine_beat_segment,
    )
except ImportError as e:
    tried = BASE_DIR.resolve()
    sys.stderr.write(
        "無法載入 analysis_engine / beat_chord_refinement。\n"
        "請設定環境變數 IGTGS_ROOT 指向含上述模組的 IGTGS 專案根目錄，\n"
        "或將 IGTGS 放在下列任一位置（已自動搜尋過）：\n"
        f"  {tried.parent / 'igtgs'}\n"
        f"  {tried.parent / 'IGTGS'}\n"
        f"  {tried / 'igtgs'}\n"
        "亦可把 analysis_engine.py、beat_chord_refinement.py 放在與本腳本相同資料夾。\n"
        f"原始錯誤：{e}\n"
    )
    raise SystemExit(1) from e

_log = logging.getLogger(__name__)

CHORD_DICT_CHOICES = ("submission", "custom", "full", "extended", "ismir2017")


def segments_to_lab_text(segments: list[dict[str, Any]]) -> str:
    """輸出與 lvcr ChordLabIO 一致：每行 start \\t end \\t label（秒，浮點）。"""
    rows: list[tuple[float, float, str]] = []
    for seg in segments:
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        label = str(seg.get("chord", "N") or "N")
        rows.append((start, end, label))
    rows.sort(key=lambda x: x[0])
    lines = [f"{a}\t{b}\t{c}" for a, b, c in rows]
    return "\n".join(lines) + ("\n" if lines else "")


def run_segment_wise_refine(
    audio_path: str,
    chord_segments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    loaded = get_refiner_model()
    weights_path: str | None = loaded[2] if loaded else None
    model = loaded[0] if loaded else None
    device = loaded[1] if loaded else None

    refined_segments: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []

    for seg in chord_segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        orig = str(seg.get("chord", "") or "")
        conf = float(seg.get("confidence", 1.0))

        entry: dict[str, Any] = {
            "start": start,
            "end": end,
            "originalChord": orig,
            "cnnLstmConfidence": conf,
        }

        root, quality = parse_root_quality(orig)
        new_seg = dict(seg)
        new_seg["chord"] = orig

        if loaded is None or model is None or device is None:
            entry["finalChord"] = orig
            entry["refined"] = False
            entry["skipReason"] = "model_unavailable"
            details.append(entry)
            refined_segments.append(new_seg)
            continue

        if root is None or quality is None or quality not in REFINE_QUALITIES:
            entry["finalChord"] = orig
            entry["refined"] = False
            entry["skipReason"] = "quality_not_target"
            details.append(entry)
            refined_segments.append(new_seg)
            continue

        result = refine_beat_segment(audio_path, start, end, model, device)
        if result is None:
            entry["finalChord"] = orig
            entry["refined"] = False
            entry["skipReason"] = "segment_load_or_infer_failed"
            details.append(entry)
            refined_segments.append(new_seg)
            continue

        best_label, ref_confidence, prob_map = result
        entry["refinerLabel"] = best_label
        entry["confidence"] = ref_confidence
        entry["probabilities"] = prob_map

        if ref_confidence < REFINER_CONFIDENCE_MIN:
            entry["finalChord"] = orig
            entry["refined"] = False
            entry["skipReason"] = "low_confidence"
            details.append(entry)
            refined_segments.append(new_seg)
            continue

        final_chord = f"{root}:{best_label}"
        new_seg["chord"] = final_chord
        entry["finalChord"] = final_chord
        entry["refined"] = True
        entry["skipReason"] = None
        details.append(entry)
        refined_segments.append(new_seg)

    return refined_segments, details, weights_path


def run_pipeline(
    audio_path: Path,
    chord_dict: str,
) -> tuple[dict[str, Any], str]:
    _log.info("分析音檔（beat + chord，dict=%s）：%s", chord_dict, audio_path)
    kw: dict[str, Any] = {
        "beat_detector": "madmom",
        "chord_detector": "LVCR",
    }
    sig = inspect.signature(analyze_audio_file)
    if "chord_dict" in sig.parameters:
        kw["chord_dict"] = chord_dict
    elif "chord_dict_name" in sig.parameters:
        kw["chord_dict_name"] = chord_dict
    else:
        kw["chord_dict"] = chord_dict
    beat_data, chord_data = analyze_audio_file(str(audio_path), **kw)

    segments_in = list(chord_data.get("chords") or [])
    duration = float(chord_data.get("duration") or beat_data.get("duration") or 0.0)
    if segments_in and duration <= 0:
        duration = float(segments_in[-1].get("end") or 0.0)

    refined_segments, segment_details, weights_path = run_segment_wise_refine(
        str(audio_path),
        segments_in,
    )

    lab_text = segments_to_lab_text(refined_segments)

    payload: dict[str, Any] = {
        "audioPath": str(audio_path),
        "duration": duration,
        "beatModel": beat_data.get("model_used") or "madmom",
        "chordModel": chord_data.get("model_used") or "LVCR",
        "chordDict": chord_data.get("chord_dict_used") or chord_data.get("chord_dict") or chord_dict,
        "chordDictRequested": chord_data.get("chord_dict_requested") or chord_dict,
        "chordDictUsed": chord_data.get("chord_dict_used") or chord_data.get("chord_dict") or chord_dict,
        "refinerWeightsPath": weights_path,
        "confidenceThreshold": REFINER_CONFIDENCE_MIN,
        "targetQualities": sorted(REFINE_QUALITIES),
        "segmentCount": len(segments_in),
        "refinedCount": sum(1 for d in segment_details if d.get("refined")),
        "chordsOriginal": segments_in,
        "chordsAfterRefine": refined_segments,
        "segmentRefineLog": segment_details,
        "labText": lab_text,
    }
    return payload, lab_text


def main() -> int:
    parser = argparse.ArgumentParser(description="和弦辨識 + ChordRefiner + 輸出 .lab")
    parser.add_argument("audio", type=Path, help="輸入音檔路徑")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="將 JSON 結果寫入檔案（未指定則印到 stdout）",
    )
    parser.add_argument(
        "--lab-out",
        type=Path,
        default=None,
        help="將 LSTM+Refiner 後的和弦軌寫成 Audacity/ChordLab 格式 .lab（Tab 分隔）",
    )
    parser.add_argument(
        "--chord-dict",
        default="submission",
        choices=CHORD_DICT_CHOICES,
        help="和弦詞表（對應 lvcr/data/{name}_chord_list.txt）",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="印出除錯 log")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    audio_path = args.audio.expanduser().resolve()
    if not audio_path.is_file():
        print(f"找不到音檔：{audio_path}", file=sys.stderr)
        return 1

    try:
        payload, lab_text = run_pipeline(audio_path, args.chord_dict)
    except Exception as e:
        _log.exception("分析失敗")
        print(f"分析失敗：{e}", file=sys.stderr)
        return 2

    # JSON 不含巨長 lab 重複（另存 --lab-out）
    payload_out = {k: v for k, v in payload.items() if k != "labText"}

    text = json.dumps(payload_out, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        _log.info("已寫入 JSON：%s", args.output.resolve())
    else:
        print(text)

    if args.lab_out:
        args.lab_out.write_text(lab_text, encoding="utf-8")
        _log.info("已寫入 LAB：%s", args.lab_out.resolve())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
