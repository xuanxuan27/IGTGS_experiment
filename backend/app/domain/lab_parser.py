"""解析 .lab 段列（與前端 parseLab 一致）。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabSegment:
    start: float
    end: float
    label: str


def parse_lab(lab_text: str) -> list[LabSegment]:
    segs: list[LabSegment] = []
    for raw in lab_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            start = float(parts[0])
            end = float(parts[1])
        except ValueError:
            continue
        label = " ".join(parts[2:])
        if not (start == start and end == end and end > start):
            continue
        segs.append(LabSegment(start=start, end=end, label=label))
    segs.sort(key=lambda s: s.start)
    return segs
