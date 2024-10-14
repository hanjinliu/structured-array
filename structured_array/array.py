from __future__ import annotations
from types import MappingProxyType
from typing import (
    Any,
    Iterator,
    Literal,
    Sequence,
    SupportsIndex,
    overload,
    TYPE_CHECKING,
)

import numpy as np
from structured_array.groupby import GroupBy
from structured_array.typing import IntoExpr, IntoIndex, IntoDType
from structured_array._normalize import into_expr_multi, basic_dtype, unstructure
from tabulate import tabulate

if TYPE_CHECKING:
    from typing import Self


class StructuredArray:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    @classmethod
    def _from_ndarray(cls, arr: np.ndarray) -> Self:
        """Create a new instance of the class."""
        return cls(arr)

    def view(self, cls: type[Self]) -> Self:
        """View the structured array as another class."""
        if not isinstance(cls, type) or not issubclass(cls, StructuredArray):
            raise TypeError("cls must be a subclass of StructuredArray")
        return cls(self._arr)

    @overload
    def to_dict(self, *, asarray: Literal[True] = True) -> dict[str, np.ndarray]: ...
    @overload
    def to_dict(self, *, asarray: Literal[False] = True) -> dict[str, list[Any]]: ...

    def to_dict(self, asarray: bool = True) -> dict[str, np.ndarray]:
        """Convert the structured array to a dictionary of columns."""
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
    def shape(self) -> tuple[int, int]:
        """Shape of the structured array."""
        return (len(self._arr), len(self.columns))

    @property
    def schema(self) -> MappingProxyType[str, np.dtype]:
        """MappingProxyType of column names to dtypes."""
        return MappingProxyType({k: v[0] for k, v in self._arr.dtype.fields.items()})

    def head(self, n: int = 5) -> Self:
        """Return the first n rows of the structured array."""
        return self._from_ndarray(self._arr[:n])

    def tail(self, n: int = 5) -> Self:
        """Return the last n rows of the structured array."""
        return self._from_ndarray(self._arr[-n:])

    def iter_rows(self) -> Iterator[np.void]:
        """Iterate over the rows of the structured array."""
        return iter(self._arr)

    def iter_columns(self) -> Iterator[np.ndarray]:
        """Iterate over the columns of the structured array."""
        return (self._arr[name] for name in self.columns)

    def rename(self, mapping: dict[str, str], **kwargs) -> Self:
        """Rename columns of the structured array."""
        mapping = {**mapping, **kwargs}
        dtypes = []
        for name in self.columns:
            dtype: np.dtype = self._arr.dtype[name]
            new_name = mapping.get(name, name)
            dtypes.append((new_name, dtype.base, dtype.shape))
        new_arr = np.empty(len(self), dtype=dtypes)
        for name in self.columns:
            new_name = mapping.get(name, name)
            arr = self._arr[name]
            new_arr[new_name] = arr
        return self._from_ndarray(new_arr)

    def filter(self, *predicates: IntoExpr) -> Self:
        """
        Filter the structured array by predicates.

        Examples
        --------
        >>> import structured_array as st
        >>> arr = st.array({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> arr.filter(st.col("a") > 1)

        a        b
        [<i8]    [<i8]
        -------  -------
        2        5
        3        6

        Multiple predicates are combined with AND.

        >>> arr.filter(st.col("a") > 1, st.col("b") < 6)

        a        b
        [<i8]    [<i8]
        -------  -------
        3        6

        """
        exprs = into_expr_multi(*predicates)
        if len(exprs) == 0:
            return self
        if len(exprs) == 1:
            predicate = exprs[0]
        else:
            predicate = exprs[0]
            for pred in exprs[1:]:
                predicate = predicate & pred
        mask = unstructure(predicate._apply_expr(self._arr))
        return self._from_ndarray(self._arr[mask])

    def group_by(
        self, by: IntoExpr | Sequence[IntoExpr], *more_by: IntoExpr
    ) -> GroupBy:
        """
        Group the structured array by columns or expressions.

        Examples
        --------
        >>> import structured_array as st
        >>> arr = st.array({"a": [1, 2, 1, 2], "b": [10, 11, 12, 13]})

        Group array by column "a".

        >>> for group, sub_arr in arr.group_by("a"):
        ...     print(repr(group))
        ...     print(repr(sub_arr))

        np.void((1,), dtype=[('a', '<i8')])
        a        b
        [<i8]    [<i8]
        -------  -------
        1        10
        1        12
        np.void((2,), dtype=[('a', '<i8')])
        a        b
        [<i8]    [<i8]
        -------  -------
        2        11
        2        13

        """
        return GroupBy.from_by(self, by, *more_by)

    def sort(self, by: IntoExpr, *, ascending: bool = True) -> Self:
        by = into_expr_multi(by)[0]
        order = np.argsort(by._apply_expr(self._arr), kind="stable")
        if not ascending:
            order = order[::-1]
        return self._from_ndarray(self._arr[order])

    def join(self, other: StructuredArray, on: str, suffix: str = "_right") -> Self:
        """Join two structured arrays on the 'uid' column"""
        from numpy.lib.recfunctions import join_by

        new_arr = join_by(on, self._arr, other._arr, r1postfix="", r2postfix=suffix)
        return self._from_ndarray(new_arr)

    def select(
        self,
        columns: IntoExpr | Sequence[IntoExpr],
        *more_columns: IntoExpr,
    ) -> Self:
        """
        Select columns from the structured array by names or expressions.

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
    ) -> Self:
        """Return a new structured array with additional columns."""
        exprs = into_expr_multi(*exprs, **named_exprs)
        arrs = [self._arr[[name]] for name in self.columns] + [
            expr._apply_expr(self._arr) for expr in exprs
        ]
        return self._new_structured_array(arrs, allow_duplicates=True)

    def to_unstructured(self, dtype=None) -> np.ndarray:
        """Return the non-structured array."""
        all_dtypes = list(set(self.dtypes))
        if len(all_dtypes) == 0:
            return self._arr.view(np.float64)
        if len(all_dtypes) > 1:
            if dtype is None:
                raise ValueError(
                    "Cannot unstructure array with multiple dtypes. Explicitly specify "
                    "`dtype` to cast the type."
                )
            out = np.empty(self.shape, dtype=dtype)
            for i, name in enumerate(self.columns):
                out[:, i] = self._arr[name].astype(dtype)
            return out
        return self._arr.view(dtype=all_dtypes[0]).reshape(self.shape)

    def __repr__(self) -> str:
        many_columns = len(self.columns) > 1

        def _repr(a, name: str):
            if isinstance(a, np.ndarray):
                return f"{a.shape!r} array"
            if isinstance(a, (str, bytes)):
                s = str(a)
                thresh = max(10, len(name) + 2)
                if len(s) > thresh and many_columns:
                    return f"{s[:thresh-1]}…"
                return s
            if isinstance(a, (int, np.integer)):
                a_str = str(a)
                if len(a_str) > 10 and many_columns:
                    return a_str[:10] + "…"
                return a_str
            return str(a)

        def _iter_short(a: np.ndarray):
            if len(a) < 10:
                yield from a
            else:
                yield from a[:5]
                yield "…"
                yield from a[-5:]

        dtype_str = [v[1] for v in self._arr.dtype.descr]
        if len(dtype_str) > 6:
            columns = [
                [_repr(_a, name) for _a in _iter_short(self[name])]
                for name in self.columns[:6]
            ]
            keys = [
                f"{name}\n[{dtype}]"
                for name, dtype in zip(self.columns[:6], dtype_str[:6])
            ]
            keys.append("…")
            columns.append(["…"] * len(columns[0]))
        else:
            columns = [
                [_repr(_a, name) for _a in _iter_short(self[name])]
                for name in self.columns
            ]
            keys = [
                f"{name}\n[{dtype}]" for name, dtype in zip(self.columns, dtype_str)
            ]
        return tabulate(
            dict(zip(keys, columns)), headers="keys", stralign="left", numalign="left"
        )

    @overload
    def __getitem__(self, key: str) -> np.ndarray: ...
    @overload
    def __getitem__(self, key: SupportsIndex) -> Self: ...
    @overload
    def __getitem__(self, key: slice | list[str] | np.ndarray) -> Self: ...
    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> Self: ...
    @overload
    def __getitem__(self, key: tuple[slice, IntoIndex]) -> np.ndarray: ...
    @overload
    def __getitem__(self, key: tuple[IntoIndex | slice]) -> np.void: ...
    @overload
    def __getitem__(self, key: tuple[IntoIndex, IntoIndex]) -> Any: ...

    def __getitem__(self, key):
        """Get a column by name, indices or slices."""
        if isinstance(key, str):
            return self._arr[key]
        elif isinstance(key, (slice, np.ndarray)):
            return self._from_ndarray(self._arr[key])
        elif isinstance(key, SupportsIndex):
            return self._from_ndarray(self._arr[key : key + 1])
        elif isinstance(key, list):
            if any(not isinstance(k, str) for k in key):
                raise TypeError("If list is given, all elements must be str")
            arrs = [self._arr[[k]] for k in key]
            return self._new_structured_array(arrs)
        elif isinstance(key, tuple):
            if len(key) == 0:
                return self
            elif len(key) == 1:
                return self[key[0]]
            elif len(key) == 2:
                r, c = key
                if isinstance(r, slice) and isinstance(c, slice):
                    columns = self.columns
                    return self._new_structured_array(
                        [self._arr[r][[cname]] for cname in columns[c]]
                    )
                elif isinstance(c, str):
                    return self._arr[c][r]
                elif isinstance(c, SupportsIndex):
                    cname = self.columns[c]
                    return _slice_np_void(self._arr[cname], r)
                return _slice_np_void(self._arr[r], c)
            else:
                raise TypeError(f"Invalid key length: {len(key)}")
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def __array__(self, dtype=None, copy: bool = False) -> np.ndarray:
        if copy:
            return np.array(self._arr, dtype=dtype)
        else:
            return np.asarray(self._arr, dtype=dtype)

    def __len__(self) -> int:
        return len(self._arr)

    def __contains__(self, key: str) -> bool:
        return key in self.columns

    def _ipython_key_completions_(self) -> list[str]:  # pragma: no cover
        return list(self.columns)

    def _new_structured_array(
        self, arrs: list[np.ndarray], allow_duplicates: bool = False
    ) -> Self:
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
        return self._from_ndarray(out)


def _dtype_of_arrays(arrs: list[np.ndarray]) -> list[IntoDType]:
    return [
        (arr.dtype.names[0], basic_dtype(arr.dtype[0]), arr.shape[1:]) for arr in arrs
    ]


def _slice_np_void(value: np.void, key) -> Any:
    if isinstance(key, slice):
        s0, s1, step = key.indices(len(value))
        rng = range(s0, s1, step)
        return np.void(
            tuple(value[i] for i in rng), dtype=[value.dtype.descr[i] for i in rng]
        )
    if isinstance(key, (int, np.integer)):
        return value[key]
    raise TypeError(f"Invalid key type: {type(key)}")
