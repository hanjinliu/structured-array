from __future__ import annotations

from typing import TYPE_CHECKING, Iterator
import numpy as np
from structured_array.typing import IntoExpr
from structured_array._normalize import into_expr_multi

if TYPE_CHECKING:
    from structured_array.array import StructuredArray


class GroupBy:
    def __init__(self, arr: StructuredArray, by: list[IntoExpr]) -> None:
        self._arr = arr
        self._by = by

    @classmethod
    def from_by(
        cls, arr: StructuredArray, by: str | list[str], *more_by: str
    ) -> GroupBy:
        return cls(arr, into_expr_multi(by, *more_by))

    def __iter__(self) -> Iterator[tuple[np.void, StructuredArray]]:
        from structured_array.array import StructuredArray

        arr_ref = self._arr.select(self._by)._arr
        unique_values = np.unique(arr_ref)
        for unique_value in unique_values:
            mask = arr_ref == unique_value
            yield unique_value, StructuredArray(self._arr._arr[mask])

    def agg(
        self,
        expr: IntoExpr,
        *more_expr: IntoExpr,
        **named_expr: IntoExpr,
    ) -> StructuredArray:
        from structured_array.array import StructuredArray

        exprs = into_expr_multi(expr, *more_expr, **named_expr)
        arrays: list[np.ndarray] = []
        for sl, sub in self:
            sub_processed = sub.select(exprs)
            height = len(sub_processed)
            dtype_all = sl.dtype.descr + sub_processed._arr.dtype.descr
            out = np.empty(height, dtype=dtype_all)
            out[list(sl.dtype.names)] = sl
            out[list(sub_processed._arr.dtype.names)] = sub_processed._arr
            arrays.append(out)
        return StructuredArray(np.concatenate(arrays, axis=0))
