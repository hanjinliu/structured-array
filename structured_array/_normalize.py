from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from structured_array.expression import Expr
    from structured_array.typing import IntoExpr
    from pandas.core.interchange.dataframe_protocol import DtypeKind


def into_expr(value: IntoExpr) -> Expr:
    from structured_array.expression import Expr, SelectExpr

    if isinstance(value, str):
        return Expr(SelectExpr(value))
    elif isinstance(value, Expr):
        return value
    else:
        raise TypeError(f"Expected str or Expr, got {type(value)}")


def into_expr_multi(
    *exprs: IntoExpr,
    **named_exprs: IntoExpr,
) -> list[Expr]:
    if len(exprs) == 1:
        if not isinstance(exprs[0], str) and hasattr(exprs[0], "__iter__"):
            exprs = exprs[0]
    named = [into_expr(col).alias(name) for name, col in named_exprs.items()]
    return [into_expr(col) for col in exprs] + named


def basic_dtype(d: np.dtype) -> np.dtype:
    if d.names is None:
        return d
    if len(d.descr) > 1:
        raise ValueError(f"More than one field in dtype: {d!r}")
    return np.dtype(d.descr[0][1])


class ColumnCaster:
    def cast(self, arr):
        return arr

    def uncast(self, arr):
        return arr


class NamedColumnCaster(ColumnCaster):
    def __init__(self, name: str, dtype) -> None:
        self.name = name
        self.dtype = dtype

    def cast(self, arr: np.ndarray):
        return arr[self.name]

    def uncast(self, arr: np.ndarray):
        if arr.ndim < 2:
            out = np.asarray(arr, dtype=[(self.name, self.dtype, ())])
        else:
            out = np.empty(arr.shape[0], dtype=[(self.name, self.dtype, arr.shape[1:])])
            out[self.name] = arr
        return out


def caster(arr, dtype=None) -> ColumnCaster:
    if arr.dtype.names is None:
        return ColumnCaster()
    return NamedColumnCaster(arr.dtype.names[0], dtype)


def unstructure(arr: np.ndarray | np.generic) -> np.ndarray:
    """Convert a structured array to a regular array."""
    if not isinstance(arr, np.ndarray) or arr.dtype.names is None:
        return arr
    return caster(arr).cast(arr)


_DTYPE_KINDS = {
    0: "i",
    1: "u",
    2: "f",
    20: "b",
    21: "S",
    22: "M",
    23: "O",
}


def dtype_kind_to_dtype(kind: tuple[DtypeKind, int, str, str]) -> np.dtype:
    _kind = _DTYPE_KINDS[kind[0].value]
    _byte = kind[1] // 8
    return np.dtype(f"{_kind}{_byte}")
