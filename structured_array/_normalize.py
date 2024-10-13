from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

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
    columns: IntoExpr | Sequence[IntoExpr],
    *more_columns: IntoExpr,
    **named_columns: IntoExpr,
) -> list[Expr]:
    if isinstance(columns, str) or not hasattr(columns, "__iter__"):
        columns = [columns]
    all_columns = [*columns, *more_columns]
    named = [into_expr(col).alias(name) for name, col in named_columns.items()]
    return [into_expr(col) for col in all_columns] + named


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

    def cast(self, arr):
        return arr[self.name]

    def uncast(self, arr):
        return np.asarray(arr, dtype=[(self.name, self.dtype, arr.shape[1:])])


def caster(arr, dtype=None) -> ColumnCaster:
    if arr.dtype.names is None:
        return ColumnCaster()
    return NamedColumnCaster(arr.dtype.names[0], dtype)


def unstructure(arr: np.ndarray) -> np.ndarray:
    """Convert a structured array to a regular array."""
    if arr.dtype.names is None:
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
