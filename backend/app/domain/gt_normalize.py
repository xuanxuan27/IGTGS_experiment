"""GT 和弦列表 capo 轉換（與前端 updateGTDisplay 一致）。"""

from __future__ import annotations

import re

NOTE_TO_PC: dict[str, int] = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "F": 5,
    "E#": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}
PC_TO_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PC_TO_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

_MAIN_CHORD = re.compile(r"^([A-G])([#b]?)(.*)$")


def norm_chord(s: str | None) -> str:
    if s is None:
        return ""
    t = (
        str(s)
        .replace(" ", "")
        .replace("－", "-")
        .replace("—", "-")
        .replace("–", "-")
        .strip()
    )
    return t


def transpose_note_name(note: str, semis: int, prefer_flat: bool = False) -> str:
    pc = NOTE_TO_PC.get(note)
    if pc is None:
        return note
    npc = (pc + semis) % 12
    return PC_TO_FLAT[npc] if prefer_flat else PC_TO_SHARP[npc]


def transpose_chord_label(label: str, semis: int) -> str:
    if not label:
        return label
    if label in ("N", "X"):
        return label
    parts = label.split("/")
    main = parts[0]
    bass = parts[1] if len(parts) > 1 else None
    m = _MAIN_CHORD.match(main)
    if not m:
        return label
    root = m.group(1) + (m.group(2) or "")
    rest = m.group(3) or ""
    prefer_flat = "b" in root
    new_root = transpose_note_name(root, semis, prefer_flat)
    out = new_root + rest
    if bass:
        mb = _MAIN_CHORD.match(bass)
        if mb:
            b_root = mb.group(1) + (mb.group(2) or "")
            b_rest = mb.group(3) or ""
            b_pref = "b" in b_root
            new_b = transpose_note_name(b_root, semis, b_pref) + b_rest
            out += "/" + new_b
        else:
            out += "/" + bass
    return out


def normalize_gt_chords(
    chords_raw: list[str],
    gt_capo: int,
    target_capo: int,
) -> list[str]:
    if not chords_raw:
        return []
    concert = [transpose_chord_label(ch, +gt_capo) for ch in chords_raw]
    return [transpose_chord_label(ch, -target_capo) for ch in concert]
