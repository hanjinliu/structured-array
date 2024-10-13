from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from structured_array._normalize import caster, unstructure


class UnitExpr(ABC):
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
    def __init__(self, ufunc, *args, **kwargs) -> None:
        self.ufunc = ufunc
        self.args = args
        self.kwargs = kwargs

    def apply(self, arr: np.ndarray) -> np.ndarray:
        _caster = caster(arr)
        return _caster.uncast(self.ufunc(_caster.cast(arr), *self.args, **self.kwargs))


class NArgExpr(UnitExpr):
    def __init__(self, ops: list[UnitExpr], func, **kwargs) -> None:
        self.ops = ops
        self.func = func
        self.kwargs = kwargs

    def apply(self, arr: np.ndarray) -> np.ndarray:
        _caster = caster(arr)
        _args = [unstructure(op.apply(arr)) for op in self.ops]
        out = _caster.uncast(self.func(*_args, **self.kwargs))
        return out


class SelectExpr(UnitExpr):
    def __init__(self, columns: list[str]) -> None:
        self.columns = list(columns)

    def apply(self, arr: np.ndarray) -> np.ndarray:
        return arr[self.columns]


class AliasExpr(UnitExpr):
    def __init__(self, alias: str) -> None:
        self.alias = alias

    def apply(self, arr: np.ndarray) -> np.ndarray:
        ar = unstructure(arr)
        return np.asarray(ar, dtype=[(self.alias, ar.dtype, arr.shape[1:])])


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
        return np.asarray(ar, dtype=[(arr.dtype.names[0], self.dtype, arr.shape[1:])])


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
