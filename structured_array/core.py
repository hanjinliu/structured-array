from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Sequence, TYPE_CHECKING, cast
import numpy as np
from structured_array._normalize import dtype_kind_to_dtype
from structured_array.array import StructuredArray
from structured_array.expression import Expr, SelectExpr, LitExpr, ArangeExpr

if TYPE_CHECKING:
    from pandas.core.interchange.dataframe_protocol import DataFrame, Column


def col(name: str | Sequence[str], *more_names: str) -> Expr:
    if isinstance(name, str):
        name = [name]
    return Expr(SelectExpr([*name, *more_names]))


def lit(value, dtype=None) -> Expr:
    return Expr(LitExpr(value, dtype))


def arange(start=None, stop=None, step=1, dtype=None) -> Expr:
    return Expr(ArangeExpr(start, stop, step, dtype))


def array(arr, schema=None) -> StructuredArray:
    """Construct a StructuredArray from any object."""
    if schema is None:
        schema = {}
    if isinstance(arr, dict):
        series = []
        dtypes = []
        heights: set[int] = set()
        for name, data in arr.items():
            data = np.asarray(data)
            series.append(data)
            dtypes.append((name, schema.get(name, data.dtype), data.shape[1:]))
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
        nrows = df.num_rows()
        dtypes = [
            (
                name,
                schema.get(
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
            arr = np.asarray(arr, dtype=schema.items())
        else:
            arr = np.asarray(arr)
        if arr.dtype.names is None:
            raise ValueError("Input array must be structured")
        return StructuredArray(arr)


def read_npy(path: str | Path | bytes) -> StructuredArray:
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
