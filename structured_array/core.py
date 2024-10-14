from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Sequence, TYPE_CHECKING, cast
import numpy as np
from structured_array._normalize import dtype_kind_to_dtype
from structured_array.array import StructuredArray
from structured_array.expression import Expr
from structured_array.expression import _unitexpr as _uexp

if TYPE_CHECKING:
    from structured_array.typing import SchemaType
    from pandas.core.interchange.dataframe_protocol import DataFrame, Column


def col(name: str | Sequence[str], *more_names: str) -> Expr:
    """Expression for a column selection."""
    if isinstance(name, str):
        name = [name]
    return Expr(_uexp.SelectExpr([*name, *more_names]))


def lit(value, dtype=None) -> Expr:
    """Literal expression."""
    return Expr(_uexp.LitExpr(value, dtype))


def arange(start=None, stop=None, step=1, dtype=None) -> Expr:
    """
    Expression for a range of values.

    Examples
    --------
    >>> import structured_array as st
    >>> arr = st.array({"a": [0, 0, 0]})
    >>> arr.with_columns(st.arange())
    a        arange
    [<i8]    [<i8]
    -------  --------
    0        0
    0        1
    0        2

    >>> arr.with_columns(b=st.arange(stop=10))
    a        b
    [<i8]    [<i8]
    -------  -------
    0        7
    0        8
    0        9
    """
    return Expr(_uexp.ArangeExpr(start, stop, step, dtype))


def linspace(start, stop, endpoint=True, dtype=None) -> Expr:
    """
    Expression for a linearly spaced range of values.

    Examples
    --------
    >>> import structured_array as st
    >>> arr = st.array({"a": [0, 0, 0]})
    >>> arr.with_columns(st.linspace(0, 1))
    a        linspace
    [<i8]    [<f8]
    -------  ----------
    0        0
    0        0.5
    0        1
    """
    return Expr(_uexp.LinspaceExpr(start, stop, endpoint, dtype))


def _schema_to_dict(s: SchemaType) -> dict[str, np.dtype]:
    if s is None:
        schema = {}
    elif isinstance(s, list) and isinstance(s[0], str):
        schema = {name: None for name in s}
    else:
        schema = dict(s)
    return schema


def array(
    arr,
    schema: SchemaType | None = None,
    schema_overrides: SchemaType | None = None,
) -> StructuredArray:
    """Construct a StructuredArray from any object."""
    schema = _schema_to_dict(schema)
    schema_overrides = _schema_to_dict(schema_overrides)
    schema_overrides.update(schema)
    if isinstance(arr, dict):
        if schema.keys() - arr.keys():
            raise ValueError("Schema keys must be a subset of input keys")
        series = []
        dtypes = []
        heights: set[int] = set()
        for name, data in arr.items():
            data = np.asarray(data)
            series.append(data)
            dtypes.append(
                (name, schema_overrides.get(name, data.dtype), data.shape[1:])
            )
            heights.add(data.shape[0])
        if len(heights) > 1:
            raise ValueError("All arrays must have the same number of rows")
        if len(dtypes) == 0:
            return StructuredArray(np.empty(0, dtype=[]))
        out = np.empty(max(heights), dtype=dtypes)
        for (dtype_name, _, _), data in zip(dtypes, series):
            out[dtype_name] = data
        return StructuredArray(out)
    elif hasattr(arr, "__dataframe__"):
        df = cast("DataFrame", arr.__dataframe__())
        if set(schema.keys()) - set(df.column_names()):
            raise ValueError("Schema keys must be a subset of input keys")
        nrows = df.num_rows()
        dtypes = [
            (
                name,
                schema_overrides.get(
                    name, dtype_kind_to_dtype(df.get_column_by_name(name).dtype)
                ),
            )
            for name in df.column_names()
        ]
        out = np.empty(nrows, dtype=dtypes)
        for name in df.column_names():
            out[name] = _column_to_numpy(df.get_column_by_name(name))
        return StructuredArray(out)
    else:
        if schema:
            # if arr is already an array, update the default dtype
            if not isinstance(arr, np.ndarray):
                arr = np.array(arr)
            for k, v in schema.items():
                if v is None:
                    if arr.dtype.names is None:
                        schema[k] = arr.dtype
                    else:
                        schema[k] = arr.dtype[k]
            arr = arr.ravel().view(dtype=list(schema.items()))
        else:
            arr = np.asarray(arr)
        if arr.dtype.names is None:
            # name column_0, column_1, etc.
            dtype = arr.dtype
            if arr.ndim == 0:
                raise ValueError("0-dimensional arrays are not supported")
            elif arr.ndim == 1:
                dtypes = [("column_0", dtype, ())]
                arr = arr.view(dtype=dtypes)
            else:
                shape_inner = arr.shape[2:]
                dtypes = [
                    (f"column_{i}", dtype, shape_inner) for i in range(arr.shape[1])
                ]
                arr = arr.ravel().view(dtype=dtypes)
        return StructuredArray(arr)


def read_npy(path: str | Path | bytes) -> StructuredArray:
    """Read a structured numpy array from a file."""
    ar = np.load(path)
    if not isinstance(ar, np.ndarray):
        raise ValueError("Input file is not a numpy array")
    return StructuredArray(ar)


def _column_to_numpy(col: Column) -> np.ndarray:
    buf = col.get_buffers()["data"][0]
    dtype = dtype_kind_to_dtype(col.dtype)
    ptr = buf.ptr
    bufsize = buf.bufsize
    ctypes_array = (ctypes.c_byte * bufsize).from_address(ptr)
    return np.frombuffer(ctypes_array, dtype=dtype)
