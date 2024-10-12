from __future__ import annotations
from typing import Sequence

import numpy as np
from structured_array.types import IntoExpr
from structured_array._normalize import into_expr_multi
from tabulate import tabulate


class StructuredArray:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert the StructuredArray to a dictionary of columns."""
        return {name: self._arr[name] for name in self.columns}

    def to_npy(self, path: str) -> None:
        np.save(path, self._arr)
        return None

    @property
    def columns(self) -> tuple[str, ...]:
        """Tuple of column names."""
        return self._arr.dtype.names

    @property
    def dtypes(self) -> list[np.dtype]:
        """List of dtypes of each column."""
        return [v[0] for v in self._arr.dtype.fields.values()]

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the StructuredArray."""
        return self._arr.shape

    def head(self, n: int = 5) -> StructuredArray:
        return StructuredArray(self._arr[:n])

    def tail(self, n: int = 5) -> StructuredArray:
        return StructuredArray(self._arr[-n:])

    def filter(self, expr: IntoExpr) -> StructuredArray:
        expr = into_expr_multi(expr)[0]
        mask = expr._apply_expr(self._arr)
        return StructuredArray(self._arr[mask])

    def sort(self, expr: IntoExpr, *, ascending: bool = True) -> StructuredArray:
        expr = into_expr_multi(expr)[0]
        order = np.argsort(expr._apply_expr(self._arr), kind="stable")
        if not ascending:
            order = order[::-1]
        return StructuredArray(self._arr[order])

    def join(
        self, other: StructuredArray, on: str, suffix: str = "_right"
    ) -> StructuredArray:
        """Join two StructuredArrays on the 'uid' column"""
        from numpy.lib.recfunctions import join_by

        new_arr = join_by(on, self._arr, other._arr, r1postfix="", r2postfix=suffix)
        return StructuredArray(new_arr)

    def select(
        self,
        columns: IntoExpr | Sequence[IntoExpr],
        *more_columns: IntoExpr,
    ) -> StructuredArray:
        """
        Select columns from the StructuredArray by names or expressions.

        >>> import structured_array as st
        >>> arr = st.array({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> arr.select("a")
        """
        exprs = into_expr_multi(columns, *more_columns)
        arrs = [expr._apply_expr(self._arr) for expr in exprs]
        return self._new_structured_array(arrs)

    def with_columns(
        self,
        columns: IntoExpr | Sequence[IntoExpr],
        *more_columns: IntoExpr,
        **named_columns: IntoExpr,
    ) -> StructuredArray:
        """Return a new StructuredArray with additional columns."""
        exprs = into_expr_multi(columns, *more_columns, **named_columns)
        arrs = [expr._apply_expr(self._arr) for expr in exprs]
        return self._new_structured_array(arrs)

    def __repr__(self) -> str:
        def _repr(a, name: str):
            if isinstance(a, np.ndarray):
                return f"{a.shape!r} array"
            if isinstance(a, (str, bytes)):
                s = str(a)
                thresh = max(16, len(name) + 2)
                if len(s) > thresh:
                    return f"{s[:thresh-1]}…"
                return s
            return str(a)

        def _iter_short(a: np.ndarray):
            if len(a) < 10:
                yield from a
            else:
                yield from a[:5]
                yield "..."
                yield from a[-5:]

        columns = [
            [_repr(_a, name) for _a in _iter_short(self[name])] for name in self.columns
        ]
        dtype_str = [v[1] for v in self._arr.dtype.descr]
        keys = [f"{name}\n[{dtype}]" for name, dtype in zip(self.columns, dtype_str)]
        return tabulate(dict(zip(keys, columns)), headers="keys")

    def __getitem__(self, key: str) -> np.ndarray:
        """Get a column by name."""
        return self._arr[key]

    def __array__(self, dtype=None, copy: bool = False) -> np.ndarray:
        if copy:
            return np.array(self._arr, dtype=dtype)
        else:
            return np.asarray(self._arr, dtype=dtype)

    def _new_structured_array(self, arrs: list[np.ndarray]) -> StructuredArray:
        dtypes = [
            (arr.dtype.names[0], arr.dtype[0].base, arr.shape[1:]) for arr in arrs
        ]
        out = np.empty(len(self._arr), dtype=dtypes)
        for name, arr in zip(self.columns, arrs):
            out[name] = arr
        return StructuredArray(out)
