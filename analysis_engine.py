from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from igtgs_paths import resolve_igtgs_backend_dir
from madmom_compat import prepare_for_madmom_import


BASE_DIR = Path(__file__).resolve().parent
IGTGS_BACKEND_DIR = resolve_igtgs_backend_dir(BASE_DIR)
FIXED_BEAT_DETECTOR = "madmom"
FIXED_CHORD_DETECTOR = "lvcr"
BACKEND_CHORD_DETECTOR = "LVCR"
CHORD_DETECTOR_ALIASES = {"lvcr", "chord-cnn-lstm", "LVCR"}
BACKEND_CHORD_DICT = "submission"
CHORD_DICT_ALIASES = {"submission", "custom", "full", "extended", "ismir2017"}


def _prepare_backend_imports() -> None:
    prepare_for_madmom_import()
    backend_dir = str(IGTGS_BACKEND_DIR)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    from compat import apply_all  # type: ignore

    # 套用相容性修正，避免 NumPy／SciPy／madmom／librosa 版本差異
    apply_all()


@lru_cache(maxsize=1)
def get_services() -> tuple[Any, Any]:
    _prepare_backend_imports()

    from beat_service import BeatDetectionService  # type: ignore
    from chord_service import ChordRecognitionService  # type: ignore

    beat_service = BeatDetectionService()
    chord_service = ChordRecognitionService()
    return beat_service, chord_service


def get_engine_status() -> dict[str, Any]:
    try:
        get_services()
        return {
            "success": True,
            "mode": "local-engine",
            "sourceDir": str(IGTGS_BACKEND_DIR),
            "availableBeatDetectors": [FIXED_BEAT_DETECTOR],
            "availableChordDetectors": sorted(CHORD_DETECTOR_ALIASES),
            "availableChordDicts": sorted(CHORD_DICT_ALIASES),
            "defaultBeatDetector": FIXED_BEAT_DETECTOR,
            "defaultChordDetector": FIXED_CHORD_DETECTOR,
            "defaultChordDict": BACKEND_CHORD_DICT,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "mode": "local-engine",
            "sourceDir": str(IGTGS_BACKEND_DIR),
            "error": str(exc),
        }


def analyze_audio_file(
    audio_path: str,
    beat_detector: str = FIXED_BEAT_DETECTOR,
    chord_detector: str = FIXED_CHORD_DETECTOR,
    chord_dict: str = "submission",
) -> tuple[dict[str, Any], dict[str, Any]]:
    if beat_detector != FIXED_BEAT_DETECTOR:
        raise RuntimeError(f"Only {FIXED_BEAT_DETECTOR} is supported")

    if chord_detector not in CHORD_DETECTOR_ALIASES:
        allowed = ", ".join(sorted(CHORD_DETECTOR_ALIASES))
        raise RuntimeError(f"Only chord detectors [{allowed}] are supported")
    chord_detector_backend = BACKEND_CHORD_DETECTOR
    if chord_dict not in CHORD_DICT_ALIASES:
        allowed = ", ".join(sorted(CHORD_DICT_ALIASES))
        raise RuntimeError(f"Only chord dictionaries [{allowed}] are supported")
    chord_dict_backend = BACKEND_CHORD_DICT

    beat_service, chord_service = get_services()

    beat_data = beat_service.detect_beats(audio_path, detector=beat_detector, force=False)
    if not beat_data.get("success"):
        raise RuntimeError(beat_data.get("error") or "Beat detection failed")

    chord_data = chord_service.recognize_chords(
        audio_path,
        detector=chord_detector_backend,
        chord_dict=chord_dict_backend,
        force=False,
    )
    if not chord_data.get("success"):
        raise RuntimeError(chord_data.get("error") or "Chord recognition failed")
    chord_data["chord_dict_requested"] = chord_dict
    chord_data["chord_dict_used"] = chord_data.get("chord_dict") or chord_dict_backend

    return beat_data, chord_data
