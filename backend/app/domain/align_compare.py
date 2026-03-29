"""Lab vs GT 對齊與錯誤統計（與前端 lab_compare 邏輯一致）。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .chord_mapping import CHORD_MAPPING
from .gt_normalize import norm_chord, normalize_gt_chords
from .lab_parser import LabSegment, parse_lab

GAP_COST = 1

_COLON_MAIN = re.compile(r"^([A-G])([#b]?)(?::)(.+)$")
_MAIN = re.compile(r"^([A-G])([#b]?)(.*)$")


def is_rest_like(label: str | None) -> bool:
    return norm_chord(label or "") == "N"


def _parse_root_suffix_bass(label: str) -> dict[str, Any] | None:
    s = norm_chord(label)
    if not s:
        return None
    if s in ("N", "X"):
        return {"special": s}
    parts = s.split("/")
    main = parts[0]
    bass = parts[1] if len(parts) > 1 else None
    cm = _COLON_MAIN.match(main)
    if cm:
        root = cm.group(1) + (cm.group(2) or "")
        suffix = cm.group(3)
    else:
        m = _MAIN.match(main)
        if not m:
            return None
        root = m.group(1) + (m.group(2) or "")
        suffix = m.group(3) or ""
    return {"root": root, "suffix": suffix, "bass": bass}


def _map_suffix_to_quality(suffix: str) -> str:
    rules = CHORD_MAPPING.get("suffix_rules") or []
    sorted_rules = sorted(rules, key=lambda r: len((r.get("in") or "")), reverse=True)
    for r in sorted_rules:
        if (suffix or "") == (r.get("in") or ""):
            return str(r["out"])
    notes = CHORD_MAPPING.get("notes") or {}
    return suffix or str(notes.get("default_quality", "maj"))


def _resolve_bass_to_notes(bass: str, root: str, quality: str) -> list[str] | None:
    try:
        deg = int(bass)
    except ValueError:
        return None
    if deg < 1 or deg > 7:
        return None
    bi = CHORD_MAPPING.get("bass_inversion") or {}
    minor_q = bi.get("minor_qualities") or []
    scale = (
        (bi.get("scale_minor") or {}).get(root)
        if quality in minor_q
        else (bi.get("scale_major") or {}).get(root)
    )
    if not scale or deg not in scale:
        return None
    note = scale[deg]
    enh = CHORD_MAPPING.get("root_enharmonics") or {}
    return list(enh.get(note, [note]))


def _chord_candidates(label: str) -> set[str]:
    info = _parse_root_suffix_bass(label)
    if not info:
        return set()
    if "special" in info:
        return {info["special"]}
    root = info["root"]
    suffix = info["suffix"]
    bass = info["bass"]
    enh = CHORD_MAPPING.get("root_enharmonics") or {}
    roots = list(enh.get(root, [root]))
    quality = _map_suffix_to_quality(suffix)
    out: set[str] = set()
    for r in roots:
        bass_list: list[str | None]
        if bass:
            try:
                deg = int(bass)
            except ValueError:
                deg = -1
            if 1 <= deg <= 7:
                bl = _resolve_bass_to_notes(bass, r, quality)
                bass_list = list(bl) if bl else [bass]
            else:
                bass_list = list(enh.get(bass, [bass]))
        else:
            bass_list = [None]
        for b in bass_list:
            out.add(f"{r}:{quality}" + (f"/{b}" if b else ""))
    return out


def chord_equals_by_mapping(a: str, b: str) -> bool:
    na, nb = norm_chord(a), norm_chord(b)
    if not na or not nb:
        return True
    if na == "N":
        return True
    ca, cb = _chord_candidates(na), _chord_candidates(nb)
    if not ca or not cb:
        return na == nb
    return not ca.isdisjoint(cb)


def needleman_wunsch(seq_a: list[str], seq_b: list[str]) -> list[dict[str, Any]]:
    n, m = len(seq_a), len(seq_b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i * GAP_COST
        trace[i][0] = "D"
    for j in range(m + 1):
        dp[0][j] = j * GAP_COST
        trace[0][j] = "I"
    trace[0][0] = None
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_cost = 0 if chord_equals_by_mapping(seq_a[i - 1], seq_b[j - 1]) else 1
            m_score = dp[i - 1][j - 1] + match_cost
            d_score = dp[i - 1][j] + GAP_COST
            i_score = dp[i][j - 1] + GAP_COST
            best = min(m_score, d_score, i_score)
            dp[i][j] = best
            if best == m_score:
                trace[i][j] = "M"
            elif best == d_score:
                trace[i][j] = "D"
            else:
                trace[i][j] = "I"
    alignment: list[dict[str, Any]] = []
    i, j = n, m
    while i > 0 or j > 0:
        t = trace[i][j]
        if t == "M":
            alignment.insert(0, {"a": i - 1, "b": j - 1, "op": "M"})
            i -= 1
            j -= 1
        elif t == "D":
            alignment.insert(0, {"a": i - 1, "b": -1, "op": "I"})
            i -= 1
        else:
            alignment.insert(0, {"a": -1, "b": j - 1, "op": "D"})
            j -= 1
    return alignment


@dataclass
class LabAlignPack:
    alignment: list[dict[str, Any]]
    lab_entries: list[dict[str, Any]]


def align_lab_to_gt(lab_segs: list[LabSegment], gt_chords: list[str]) -> LabAlignPack:
    lab_entries = [
        {"seg_idx": i, "label": s.label}
        for i, s in enumerate(lab_segs)
        if not is_rest_like(s.label)
    ]
    lab_labels = [e["label"] for e in lab_entries]
    if not lab_labels or not gt_chords:
        return LabAlignPack(alignment=[], lab_entries=lab_entries)
    alignment = needleman_wunsch(lab_labels, gt_chords)
    return LabAlignPack(alignment=alignment, lab_entries=lab_entries)


def alignment_to_maps(pack: LabAlignPack) -> dict[str, Any]:
    seg_idx_to_result: dict[int, dict[str, Any]] = {}
    gt_missing: set[int] = set()
    lab_chord_idx_to_seg_idx = {i: e["seg_idx"] for i, e in enumerate(pack.lab_entries)}
    for step in pack.alignment:
        a, b, op = step["a"], step["b"], step["op"]
        if op == "M":
            seg_idx = lab_chord_idx_to_seg_idx.get(a)
            if seg_idx is not None:
                seg_idx_to_result[seg_idx] = {"gt_idx": b, "op": "M"}
        elif op == "I":
            seg_idx = lab_chord_idx_to_seg_idx.get(a)
            if seg_idx is not None:
                seg_idx_to_result[seg_idx] = {"gt_idx": -1, "op": "I"}
        elif op == "D":
            gt_missing.add(b)
    return {"seg_idx_to_result": seg_idx_to_result, "gt_missing": gt_missing}


@dataclass
class MergeCtx:
    before_seg: list[LabSegment]
    after_seg: list[LabSegment]
    gt_disp: list[str]
    before_align: LabAlignPack | None
    after_align: LabAlignPack | None
    before_maps: dict[str, Any] | None
    after_maps: dict[str, Any] | None


def _rebuild_alignments(ctx: MergeCtx) -> None:
    ctx.before_align = ctx.after_align = None
    ctx.before_maps = ctx.after_maps = None
    if ctx.gt_disp:
        if ctx.before_seg:
            p = align_lab_to_gt(ctx.before_seg, ctx.gt_disp)
            ctx.before_align = p
            ctx.before_maps = alignment_to_maps(p)
        if ctx.after_seg:
            p = align_lab_to_gt(ctx.after_seg, ctx.gt_disp)
            ctx.after_align = p
            ctx.after_maps = alignment_to_maps(p)
    elif ctx.before_seg and ctx.after_seg:
        before_labels = [s.label for s in ctx.before_seg if not is_rest_like(s.label)]
        if before_labels:
            p = align_lab_to_gt(ctx.after_seg, before_labels)
            ctx.after_align = p
            ctx.after_maps = alignment_to_maps(p)


def build_merged_rows(ctx: MergeCtx) -> list[dict[str, Any]]:
    before_seg = ctx.before_seg
    gt = ctx.gt_disp
    merged: list[dict[str, Any]] = []
    if not before_seg:
        if gt:
            for i, ch in enumerate(gt):
                merged.append(
                    {
                        "before_seg": None,
                        "gt_chord": ch,
                        "gt_idx": i,
                        "before_blank": True,
                        "gt_blank": False,
                        "before_seg_idx": -1,
                        "align_op": None,
                        "after_seg": None,
                    }
                )
        elif ctx.after_seg:
            for i, s in enumerate(ctx.after_seg):
                merged.append(
                    {
                        "before_seg": None,
                        "gt_chord": "",
                        "gt_idx": -1,
                        "before_blank": True,
                        "gt_blank": True,
                        "before_seg_idx": -1,
                        "align_op": None,
                        "after_seg": s,
                        "after_seg_idx": i,
                    }
                )
        return merged
    if not gt:
        for i, s in enumerate(before_seg):
            merged.append(
                {
                    "before_seg": s,
                    "gt_chord": "N" if is_rest_like(s.label) else "",
                    "gt_idx": -1,
                    "before_blank": False,
                    "gt_blank": True,
                    "before_seg_idx": i,
                    "align_op": None,
                    "after_seg": None,
                }
            )
        return merged
    align = ctx.before_align
    if align is None:
        return merged
    if not align.alignment:
        g = 0
        for i, s in enumerate(before_seg):
            if is_rest_like(s.label):
                merged.append(
                    {
                        "before_seg": s,
                        "gt_chord": "N",
                        "gt_idx": -1,
                        "before_blank": False,
                        "gt_blank": False,
                        "before_seg_idx": i,
                        "align_op": None,
                        "after_seg": None,
                    }
                )
            else:
                merged.append(
                    {
                        "before_seg": s,
                        "gt_chord": gt[g] if g < len(gt) else "",
                        "gt_idx": g if g < len(gt) else -1,
                        "before_blank": False,
                        "gt_blank": False,
                        "before_seg_idx": i,
                        "align_op": None,
                        "after_seg": None,
                    }
                )
                g += 1
        return merged
    lab_chord_idx_to_seg_idx = {i: e["seg_idx"] for i, e in enumerate(align.lab_entries)}
    prev_seg_idx = -1
    for step in align.alignment:
        a, b, op = step["a"], step["b"], step["op"]
        if op == "M":
            seg_idx = lab_chord_idx_to_seg_idx[a]
            for i in range(prev_seg_idx + 1, seg_idx):
                if is_rest_like(before_seg[i].label):
                    merged.append(
                        {
                            "before_seg": before_seg[i],
                            "gt_chord": "N",
                            "gt_idx": -1,
                            "before_blank": False,
                            "gt_blank": False,
                            "before_seg_idx": i,
                            "align_op": None,
                            "after_seg": None,
                        }
                    )
            merged.append(
                {
                    "before_seg": before_seg[seg_idx],
                    "gt_chord": gt[b],
                    "gt_idx": b,
                    "before_blank": False,
                    "gt_blank": False,
                    "before_seg_idx": seg_idx,
                    "align_op": "M",
                    "after_seg": None,
                }
            )
            prev_seg_idx = seg_idx
        elif op == "I":
            seg_idx = lab_chord_idx_to_seg_idx[a]
            for i in range(prev_seg_idx + 1, seg_idx):
                if is_rest_like(before_seg[i].label):
                    merged.append(
                        {
                            "before_seg": before_seg[i],
                            "gt_chord": "N",
                            "gt_idx": -1,
                            "before_blank": False,
                            "gt_blank": False,
                            "before_seg_idx": i,
                            "align_op": None,
                            "after_seg": None,
                        }
                    )
            merged.append(
                {
                    "before_seg": before_seg[seg_idx],
                    "gt_chord": "",
                    "gt_idx": -1,
                    "before_blank": False,
                    "gt_blank": True,
                    "before_seg_idx": seg_idx,
                    "align_op": "I",
                    "after_seg": None,
                }
            )
            prev_seg_idx = seg_idx
        elif op == "D":
            merged.append(
                {
                    "before_seg": None,
                    "gt_chord": gt[b],
                    "gt_idx": b,
                    "before_blank": True,
                    "gt_blank": False,
                    "before_seg_idx": -1,
                    "align_op": "D",
                    "after_seg": None,
                }
            )
    for i in range(prev_seg_idx + 1, len(before_seg)):
        if is_rest_like(before_seg[i].label):
            merged.append(
                {
                    "before_seg": before_seg[i],
                    "gt_chord": "N",
                    "gt_idx": -1,
                    "before_blank": False,
                    "gt_blank": False,
                    "before_seg_idx": i,
                    "align_op": None,
                    "after_seg": None,
                }
            )
    return merged


def _before_seg_idx_to_chord_idx(before_seg: list[LabSegment]) -> dict[int, int]:
    m: dict[int, int] = {}
    c = 0
    for i, s in enumerate(before_seg):
        if not is_rest_like(s.label):
            m[i] = c
            c += 1
    return m


def _gt_idx_to_after_seg_idx(after_maps: dict[str, Any] | None) -> dict[int, int]:
    out: dict[int, int] = {}
    if not after_maps:
        return out
    for seg_idx_str, res in (after_maps.get("seg_idx_to_result") or {}).items():
        if res.get("op") == "M" and int(res.get("gt_idx", -1)) >= 0:
            out[int(res["gt_idx"])] = int(seg_idx_str)
    return out


def _resolve_after_for_row(
    r: dict[str, Any],
    after_seg: list[LabSegment],
    ref_idx_to_after: dict[int, int],
    before_idx_map: dict[int, int],
) -> LabSegment | None:
    if r.get("after_seg"):
        return r["after_seg"]
    gt_idx = r.get("gt_idx", -1)
    bsi = r.get("before_seg_idx", -1)
    ref = gt_idx if gt_idx >= 0 else (before_idx_map.get(bsi) if bsi is not None and bsi >= 0 else None)
    if ref is None:
        return None
    a_idx = ref_idx_to_after.get(ref)
    if a_idx is None:
        return None
    return after_seg[a_idx] if a_idx < len(after_seg) else None


def _compute_row_times(merged: list[dict[str, Any]]) -> None:
    prev_end = 0.0
    for idx, r in enumerate(merged):
        bs = r.get("before_seg")
        ae = r.get("after_seg") or (r.get("_after_resolved"))
        if bs:
            r["row_start"], r["row_end"] = bs.start, bs.end
            prev_end = bs.end
        elif ae:
            r["row_start"], r["row_end"] = ae.start, ae.end
            prev_end = ae.end
        else:
            nxt = next(
                (x for x in merged[idx + 1 :] if x.get("before_seg") or x.get("_after_resolved")),
                None,
            )
            rs = prev_end
            if nxt:
                ns = nxt.get("before_seg") or nxt.get("_after_resolved")
                re = ns.start if ns else prev_end + 0.01
            else:
                re = prev_end + 0.01
            r["row_start"], r["row_end"] = rs, re


def _style_merged(
    merged: list[dict[str, Any]],
    gt_disp: list[str],
    after_seg: list[LabSegment],
    after_maps: dict[str, Any] | None,
    before_seg: list[LabSegment],
) -> None:
    ref_idx_to_after = _gt_idx_to_after_seg_idx(after_maps)
    before_idx_map = _before_seg_idx_to_chord_idx(before_seg)
    for r in merged:
        r["_after_resolved"] = _resolve_after_for_row(r, after_seg, ref_idx_to_after, before_idx_map)
    _compute_row_times(merged)
    gt_to_after = _gt_idx_to_after_seg_idx(after_maps)
    for r in merged:
        gti = r.get("gt_idx", -1)
        ao = r.get("align_op")
        bs = r.get("before_seg")
        bb = r.get("before_blank")
        gc = r.get("gt_chord") or ""
        r["before_label_extra"] = ao == "I"
        r["gt_cell_missing"] = bool(bb and gc)
        r["before_label_diff"] = False
        r["after_label_diff"] = False
        r["after_cell_blank"] = False
        ares = r.get("_after_resolved")
        bsi = r.get("before_seg_idx", -1)
        if not gt_disp:
            if not ares and (gti >= 0 or bsi >= 0):
                r["after_cell_blank"] = True
            continue
        if ao == "I" and bs:
            pass
        elif gti >= 0:
            if bs and not is_rest_like(bs.label) and not chord_equals_by_mapping(bs.label, gt_disp[gti]):
                r["before_label_diff"] = True
            a_idx = gt_to_after.get(gti)
            if a_idx is not None and a_idx < len(after_seg):
                alab = after_seg[a_idx].label or ""
                if alab and not chord_equals_by_mapping(alab, gt_disp[gti]):
                    r["after_label_diff"] = True
        if not ares and (gti >= 0 or bsi >= 0):
            r["after_cell_blank"] = True


def _compute_counts(
    merged: list[dict[str, Any]],
    gt_disp: list[str],
    after_seg: list[LabSegment],
    after_maps: dict[str, Any] | None,
    before_seg: list[LabSegment],
) -> dict[str, dict[str, int]]:
    before = {"extra": 0, "missing": 0, "wrong": 0}
    after = {"extra": 0, "missing": 0, "wrong": 0}
    if not gt_disp:
        return {
            "before": {**before, "total": 0},
            "after": {**after, "total": 0},
        }
    gt_to_after = _gt_idx_to_after_seg_idx(after_maps)
    for r in merged:
        if r.get("align_op") == "I":
            before["extra"] += 1
        if r.get("before_blank") and r.get("gt_chord"):
            before["missing"] += 1
        gti = r.get("gt_idx", -1)
        bs = r.get("before_seg")
        if gti >= 0 and bs and not is_rest_like(bs.label) and not chord_equals_by_mapping(
            bs.label, gt_disp[gti]
        ):
            before["wrong"] += 1
        a_idx = gt_to_after.get(gti) if gti >= 0 else None
        if gti >= 0 and a_idx is None:
            after["missing"] += 1
        if gti >= 0 and a_idx is not None and a_idx < len(after_seg):
            alab = after_seg[a_idx].label or ""
            if alab and not chord_equals_by_mapping(alab, gt_disp[gti]):
                after["wrong"] += 1
    after_extra = 0
    if after_maps and after_maps.get("seg_idx_to_result"):
        after_extra = sum(
            1 for res in after_maps["seg_idx_to_result"].values() if res.get("op") == "I"
        )
    after["extra"] = after_extra
    return {
        "before": {**before, "total": before["extra"] + before["missing"] + before["wrong"]},
        "after": {**after, "total": after["extra"] + after["missing"] + after["wrong"]},
    }


def _merged_to_api_row(r: dict[str, Any], idx: int) -> dict[str, Any]:
    bs = r.get("before_seg")
    ar = r.get("_after_resolved")
    gti = r.get("gt_idx", -1)
    gc = r.get("gt_chord") or ""
    if r.get("before_blank") and gc:
        gt_display = gc + " (漏標)"
    elif gc:
        gt_display = gc
    else:
        gt_display = "—"
    before_time = (
        f"{bs.start:.2f} - {bs.end:.2f}  "
        if bs
        else (f"{ar.start:.2f} - {ar.end:.2f}  " if ar else "—  ")
    )
    after_time = (
        f"{ar.start:.2f} - {ar.end:.2f}  "
        if ar
        else (f"{bs.start:.2f} - {bs.end:.2f}  " if bs else "—  ")
    )
    return {
        "idx": idx,
        "gt_idx": gti,
        "gt_chord": gt_display,
        "gt_blank": bool(r.get("gt_blank")),
        "before_blank": bool(r.get("before_blank")),
        "before_seg": {"start": bs.start, "end": bs.end, "label": bs.label} if bs else None,
        "after_seg": {"start": ar.start, "end": ar.end, "label": ar.label} if ar else None,
        "align_op": r.get("align_op"),
        "before_seg_idx": r.get("before_seg_idx", -1),
        "row_start": r.get("row_start", 0.0),
        "row_end": r.get("row_end", 0.0),
        "before_time": before_time,
        "after_time": after_time,
        "before_label_extra": bool(r.get("before_label_extra")),
        "before_label_diff": bool(r.get("before_label_diff")),
        "after_label_diff": bool(r.get("after_label_diff")),
        "after_cell_blank": bool(r.get("after_cell_blank")),
        "gt_cell_missing": bool(r.get("gt_cell_missing")),
    }


def run_compare(
    before_lab_text: str,
    after_lab_text: str,
    chords_raw: list[str],
    gt_capo: int,
    target_capo: int,
) -> dict[str, Any]:
    before_seg = parse_lab(before_lab_text) if before_lab_text.strip() else []
    after_seg = parse_lab(after_lab_text) if after_lab_text.strip() else []
    gt_disp = normalize_gt_chords(chords_raw, gt_capo, target_capo) if chords_raw else []

    ctx = MergeCtx(
        before_seg=before_seg,
        after_seg=after_seg,
        gt_disp=gt_disp,
        before_align=None,
        after_align=None,
        before_maps=None,
        after_maps=None,
    )
    _rebuild_alignments(ctx)
    merged = build_merged_rows(ctx)
    _style_merged(merged, gt_disp, after_seg, ctx.after_maps, before_seg)
    counts = _compute_counts(merged, gt_disp, after_seg, ctx.after_maps, before_seg)
    api_rows = [_merged_to_api_row(r, i) for i, r in enumerate(merged)]
    return {
        "gt_chords_disp": gt_disp,
        "merged_rows": api_rows,
        "counts": counts,
    }
