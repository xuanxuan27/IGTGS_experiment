"""
讓 PyPI 版 madmom 在 Python 3.10+ 與 NumPy 1.24+ 可 import。

- collections.MutableSequence（3.10+）
- np.float / np.int / np.bool 等（1.24+ 已移除的別名）

须在 `import madmom` 之前呼叫 `prepare_for_madmom_import()`。
"""
from __future__ import annotations


def ensure_collections_mutable_sequence() -> None:
    import collections
    import collections.abc

    if not hasattr(collections, "MutableSequence"):
        collections.MutableSequence = collections.abc.MutableSequence


def ensure_numpy_legacy_aliases_for_madmom() -> None:
    """madmom 仍使用 np.float 等；NumPy 1.24+ 已移除這些別名。"""
    import numpy as np

    # 用 __dict__ 判斷，避免 hasattr(np, "bool") 觸發 NumPy 的 FutureWarning
    d = np.__dict__
    if "float" not in d:
        np.float = np.float64  # type: ignore[attr-defined]
    if "int" not in d:
        np.int = np.int_  # type: ignore[attr-defined]
    if "bool" not in d:
        np.bool = np.bool_  # type: ignore[attr-defined]
    if "complex" not in d:
        np.complex = np.complex128  # type: ignore[attr-defined]
    if "str" not in d:
        np.str = np.str_  # type: ignore[attr-defined]
    if "long" not in d:
        np.long = np.int_  # type: ignore[attr-defined]


def prepare_for_madmom_import() -> None:
    ensure_numpy_legacy_aliases_for_madmom()
    ensure_collections_mutable_sequence()
