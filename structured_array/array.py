from __future__ import annotations
from types import MappingProxyType
from typing import Any, Iterator, Literal, Sequence, SupportsIndex, overload

import numpy as np
from structured_array.groupby import GroupBy
from structured_array.typing import IntoExpr, IntoIndex, IntoDType
from structured_array._normalize import into_expr_multi, basic_dtype, unstructure
from tabulate import tabulate


class StructuredArray:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    @overload
    def to_dict(self, *, asarray: Literal[True] = True) -> dict[str, np.ndarray]: ...
    @overload
    def to_dict(self, *, asarray: Literal[False] = True) -> dict[str, list[Any]]: ...

    def to_dict(self, asarray: bool = True) -> dict[str, np.ndarray]:
        """Convert the StructuredArray to a dictionary of columns."""
        if asarray:
            return {name: self._arr[name] for name in self.columns}
        else:
            return {name: self._arr[name].tolist() for name in self.columns}

    def write_npy(self, path: str) -> None:
        np.save(path, self._arr)
        return None

    @property
    def columns(self) -> tuple[str, ...]:
        """Tuple of column names."""
        return self._arr.dtype.names

    @property
    def dtypes(self) -> list[np.dtype]:
        """List of dtypes of each column."""
        return list(self.schema.values())

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the StructuredArray."""
        return self._arr.shape

    @property
    def schema(self) -> MappingProxyType[str, np.dtype]:
        """MappingProxyType of column names to dtypes."""
        return MappingProxyType({k: v[0] for k, v in self._arr.dtype.fields.items()})

    def head(self, n: int = 5) -> StructuredArray:
        """Return the first n rows of the StructuredArray."""
        return StructuredArray(self._arr[:n])

    def tail(self, n: int = 5) -> StructuredArray:
        """Return the last n rows of the StructuredArray."""
        return StructuredArray(self._arr[-n:])

    def iter_rows(self) -> Iterator[np.void]:
        """Iterate over the rows of the StructuredArray."""
        return iter(self._arr)

    def iter_columns(self) -> Iterator[np.ndarray]:
        """Iterate over the columns of the StructuredArray."""
        return (self._arr[name] for name in self.columns)

    def filter(self, predicate: IntoExpr) -> StructuredArray:
        predicate = into_expr_multi(predicate)[0]
        mask = predicate._apply_expr(self._arr)
        return StructuredArray(self._arr[mask])

    def group_by(
        self, by: IntoExpr | Sequence[IntoExpr], *more_by: IntoExpr
    ) -> GroupBy:
        return GroupBy.from_by(self, by, *more_by)

    def sort(self, by: IntoExpr, *, ascending: bool = True) -> StructuredArray:
        by = into_expr_multi(by)[0]
        order = np.argsort(by._apply_expr(self._arr), kind="stable")
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
        *exprs: IntoExpr | Sequence[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> StructuredArray:
        """Return a new StructuredArray with additional columns."""
        exprs = into_expr_multi(*exprs, **named_exprs)
        arrs = [self._arr[[name]] for name in self.columns] + [
            expr._apply_expr(self._arr) for expr in exprs
        ]
        return self._new_structured_array(arrs, allow_duplicates=True)

    def __repr__(self) -> str:
        def _repr(a, name: str):
            if isinstance(a, np.ndarray):
                return f"{a.shape!r} array"
            if isinstance(a, (str, bytes)):
                s = str(a)
                thresh = max(10, len(name) + 2)
                if len(s) > thresh:
                    return f"{s[:thresh-1]}…"
                return s
            return str(a)

        def _iter_short(a: np.ndarray):
            if len(a) < 10:
                yield from a
            else:
                yield from a[:5]
                yield "…"
                yield from a[-5:]

        columns = [
            [_repr(_a, name) for _a in _iter_short(self[name])] for name in self.columns
        ]
        dtype_str = [v[1] for v in self._arr.dtype.descr]
        keys = [f"{name}\n[{dtype}]" for name, dtype in zip(self.columns, dtype_str)]
        return tabulate(
            dict(zip(keys, columns)), headers="keys", stralign="left", numalign="left"
        )

    @overload
    def __getitem__(self, key: str) -> np.ndarray: ...
    @overload
    def __getitem__(self, key: SupportsIndex) -> np.void: ...
    @overload
    def __getitem__(self, key: slice | list[str] | np.ndarray) -> StructuredArray: ...
    @overload
    def __getitem__(self, key: tuple[IntoIndex | slice, IntoIndex | slice]) -> Any: ...

    def __getitem__(self, key):
        """Get a column by name, indices or slices."""
        if isinstance(key, str):
            return self._arr[key]
        elif isinstance(key, (slice, np.ndarray)):
            return StructuredArray(self._arr[key])
        elif isinstance(key, SupportsIndex):
            return StructuredArray(self._arr[key : key + 1])
        elif isinstance(key, list):
            if any(not isinstance(k, str) for k in key):
                raise TypeError("If list is given, all elements must be str")
            arrs = [self._arr[k] for k in key]
            return self._new_structured_array(arrs)
        elif isinstance(key, tuple):
            if len(key) == 0:
                return self
            elif len(key) == 1:
                return self[key[0]]
            elif len(key) == 2:
                return self._arr[key]
            else:
                raise TypeError(f"Invalid key length: {len(key)}")
        raise TypeError(f"Invalid key type: {type(key)}")

    def __array__(self, dtype=None, copy: bool = False) -> np.ndarray:
        if copy:
            return np.array(self._arr, dtype=dtype)
        else:
            return np.asarray(self._arr, dtype=dtype)

    def _new_structured_array(
        self, arrs: list[np.ndarray], allow_duplicates: bool = False
    ) -> StructuredArray:
        height = max(arr.shape[0] if arr.ndim > 0 else 1 for arr in arrs)
        if allow_duplicates:
            dtypes_all = _dtype_of_arrays(arrs)
            columns: dict[str, np.ndarray] = {}
            dtypes_dict: dict[str, IntoDType] = {}
            for (name, dtype, shape), arr in zip(dtypes_all, arrs):
                columns[name] = arr
                dtypes_dict[name] = (name, dtype, shape)
            dtypes = list(dtypes_dict.values())
            columns = {name: arr for (name, _, _), arr in zip(dtypes_all, arrs)}.items()
        else:
            dtypes = _dtype_of_arrays(arrs)
            columns = [(name, arr) for (name, _, _), arr in zip(dtypes, arrs)]
        out = np.empty(height, dtype=dtypes)
        for name, arr in columns:
            out[name] = unstructure(arr)
        return StructuredArray(out)


def _dtype_of_arrays(arrs: list[np.ndarray]) -> list[IntoDType]:
    return [
        (arr.dtype.names[0], basic_dtype(arr.dtype[0]), arr.shape[1:]) for arr in arrs
    ]
