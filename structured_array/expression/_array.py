from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from structured_array._normalize import caster
from structured_array.typing import AxisType
from structured_array.expression._namespace import ExprNamespace
from structured_array.expression import _unitexpr as _uexp

if TYPE_CHECKING:
    from structured_array.expression import Expr


class ArrayUfuncExpr(_uexp.UfuncExpr):
    def apply(self, arr: np.ndarray) -> np.ndarray:
        _caster = caster(arr)
        ar0 = _caster.cast(arr)
        kwargs = self.kwargs.copy()
        if (axis := kwargs.get("axis")) is not None:
            if isinstance(axis, (int, np.integer)):
                axis = (axis,)
            kwargs["axis"] = tuple(a + 1 if a >= 0 else a + ar0.ndim for a in axis)
        else:
            kwargs["axis"] = tuple(range(1, ar0.ndim))
        out = self.ufunc(ar0, *self.args, **kwargs)
        return _caster.uncast(out)


class SingleAxisArrayUfuncExpr(_uexp.UfuncExpr):
    def apply(self, arr: np.ndarray) -> np.ndarray:
        _caster = caster(arr)
        ar0 = _caster.cast(arr)
        kwargs = self.kwargs.copy()
        if (axis := kwargs.get("axis")) is not None:
            if not isinstance(axis, (int, np.integer)):  # pragma: no cover
                raise ValueError("`axis` must be a single integer")
            kwargs["axis"] = axis + 1 if axis >= 0 else axis + ar0.ndim
        else:
            kwargs["axis"] = 1
        out = self.ufunc(ar0, *self.args, **kwargs)
        return _caster.uncast(out)


class ArrayNamespace(ExprNamespace):
    def __getitem__(self, key) -> Expr:
        if isinstance(key, tuple):
            _key = (slice(None), *key)
        else:
            _key = (slice(None), key)
        return self._expr().__getitem__(_key)

    def min(self, axis: AxisType = None) -> Expr:
        return self._new(self._op().compose(ArrayUfuncExpr(np.min, axis=axis)))

    def max(self, axis: AxisType = None) -> Expr:
        return self._new(self._op().compose(ArrayUfuncExpr(np.max, axis=axis)))

    def sum(self, axis: AxisType = None) -> Expr:
        return self._new(self._op().compose(ArrayUfuncExpr(np.sum, axis=axis)))

    def mean(self, axis: AxisType = None) -> Expr:
        return self._new(self._op().compose(ArrayUfuncExpr(np.mean, axis=axis)))

    def std(self, axis: AxisType = None, ddof: int = 0) -> Expr:
        return self._new(
            self._op().compose(ArrayUfuncExpr(np.std, axis=axis, ddof=ddof))
        )

    def var(self, axis: AxisType = None, ddof: int = 0) -> Expr:
        return self._new(
            self._op().compose(ArrayUfuncExpr(np.var, axis=axis, ddof=ddof))
        )

    def median(self, axis: AxisType = None) -> Expr:
        return self._new(self._op().compose(ArrayUfuncExpr(np.median, axis=axis)))

    def percentile(self, q, axis: AxisType = None) -> Expr:
        return self._new(
            self._op().compose(ArrayUfuncExpr(np.percentile, q, axis=axis))
        )

    def quantile(self, q, axis: AxisType = None) -> Expr:
        return self._new(self._op().compose(ArrayUfuncExpr(np.quantile, q, axis=axis)))

    def all(self, axis: AxisType = None) -> Expr:
        return self._new(self._op().compose(ArrayUfuncExpr(np.all, axis=axis)))

    def any(self, axis: AxisType = None) -> Expr:
        return self._new(self._op().compose(ArrayUfuncExpr(np.any, axis=axis)))

    def argmin(self, axis: AxisType = None) -> Expr:
        return self._new(
            self._op().compose(SingleAxisArrayUfuncExpr(np.argmin, axis=axis))
        )

    def argmax(self, axis: AxisType = None) -> Expr:
        return self._new(
            self._op().compose(SingleAxisArrayUfuncExpr(np.argmax, axis=axis))
        )
