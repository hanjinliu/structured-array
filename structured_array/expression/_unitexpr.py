from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

from structured_array._normalize import caster, unstructure, asarray_maybe_nd


class UnitExpr(ABC):
    """Expression object that represents a single operation"""

    @abstractmethod
    def apply(self, arr: np.ndarray) -> np.ndarray:
        """Evaluate the expression on the input array"""

    def compose(self, other: UnitExpr) -> UnitExpr:
        if isinstance(other, CompositeExpr):
            return CompositeExpr([self, *other.ops])
        return CompositeExpr([self, other])


class CompositeExpr(UnitExpr):
    def __init__(self, ops: list[UnitExpr]) -> None:
        self.ops = ops

    def apply(self, arr: np.ndarray) -> np.ndarray:
        for op in self.ops:
            arr = op.apply(arr)
        return arr

    def compose(self, other: UnitExpr) -> UnitExpr:
        if isinstance(other, CompositeExpr):
            return CompositeExpr([*self.ops, *other.ops])
        return CompositeExpr([*self.ops, other])


class UfuncExpr(UnitExpr):
    def __init__(
        self,
        ufunc: Callable[..., np.ndarray],
        *args,
        **kwargs,
    ) -> None:
        self.ufunc = ufunc
        self.args = args
        self.kwargs = kwargs

    def apply(self, arr: np.ndarray) -> np.ndarray:
        _caster = caster(arr)
        return _caster.uncast(self.ufunc(_caster.cast(arr), *self.args, **self.kwargs))

    @classmethod
    def from_axis(cls, ufunc: Callable[..., np.ndarray], axis=None, **kwargs):
        if axis not in (None, 0):
            raise ValueError("`axis` must be None or 0")
        keepdims = axis is not None
        return cls(ufunc, axis=axis, keepdims=keepdims, **kwargs)


class NArgExpr(UnitExpr):
    def __init__(self, ops: list[UnitExpr], func, **kwargs) -> None:
        self.ops = ops
        self.func = func
        self.kwargs = kwargs

    def apply(self, arr: np.ndarray) -> np.ndarray:
        _arrs_structured = [op.apply(arr) for op in self.ops]
        _args = [unstructure(_a) for _a in _arrs_structured]
        _caster = caster(_arrs_structured[0])  # inherits name from the 1st arg
        out = _caster.uncast(self.func(*_args, **self.kwargs))
        return out


class SelectExpr(UnitExpr):
    def __init__(self, columns: list[str]) -> None:
        self.columns = list(columns)

    def apply(self, arr: np.ndarray) -> np.ndarray:
        if len(self.columns) == 1:
            name = self.columns[0]
            ar0 = arr[name]
            if ar0.dtype.names is not None:  # pragma: no cover
                raise RuntimeError(f"Unexpected nested structured array: {ar0!r}")
            return asarray_maybe_nd(ar0, name, ar0.dtype)
        return arr[self.columns]


class AliasExpr(UnitExpr):
    def __init__(self, alias: str) -> None:
        self.alias = alias

    def apply(self, arr: np.ndarray) -> np.ndarray:
        ar = unstructure(arr)
        if ar.dtype.names is not None:  # pragma: no cover
            raise RuntimeError(f"Unexpected nested structured array: {ar!r}")
        return asarray_maybe_nd(ar, self.alias, ar.dtype)


class LitExpr(UnitExpr):
    def __init__(self, value, dtype=None) -> None:
        self.value = value
        self.dtype = dtype

    def apply(self, arr: np.ndarray) -> np.ndarray:
        return np.full((), self.value, dtype=self.dtype).item()


class CastExpr(UnitExpr):
    def __init__(self, dtype) -> None:
        self.dtype = dtype

    def apply(self, arr: np.ndarray) -> np.ndarray:
        ar = unstructure(arr)
        return asarray_maybe_nd(ar, arr.dtype.names[0], self.dtype)


class ArangeExpr(UnitExpr):
    def __init__(self, start=None, stop=None, step=1, dtype=None) -> None:
        self.start = start
        self.stop = stop
        self.step = step
        self.dtype = dtype

    def apply(self, arr: np.ndarray) -> np.ndarray:
        num = arr.shape[0]
        if self.start is None:
            if self.stop is None:
                start = 0
            else:
                start = self.stop - num * self.step
        else:
            start = self.start
        if self.stop is None:
            stop = self.start + num * self.step
        else:
            stop = self.stop
        out = np.arange(start, stop, self.step, dtype=self.dtype)
        if out.size != num:
            raise ValueError("Size mismatch")
        return out
