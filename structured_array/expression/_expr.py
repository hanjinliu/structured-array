from __future__ import annotations
import operator
from typing import Any, Literal, Sequence

import numpy as np
from structured_array.expression import _unitexpr as _uexp
from structured_array.expression._array import ArrayNamespace
from structured_array.expression._char import StrNamespace
from structured_array.typing import IntoExpr


class Expr:
    def __init__(self, op: _uexp.UnitExpr) -> None:
        self._op = op

    def _apply_expr(self, arr: np.ndarray) -> np.ndarray:
        return self._op.apply(arr)

    arr = ArrayNamespace()
    """Namespace for element-wise operations on arrays"""

    str = StrNamespace()
    """Namespace for string operations"""

    def alias(self, alias: str) -> Expr:
        """Expression that change the column name."""
        return Expr(self._op.compose(_uexp.AliasExpr(alias)))

    def cast(self, dtype) -> Expr:
        return Expr(self._op.compose(_uexp.CastExpr(dtype)))

    def __neg__(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(operator.neg)))

    def __pos__(self) -> Expr:
        return self

    def __and__(self, other: Expr) -> Expr:
        return Expr(_uexp.NArgExpr([self._op, _to_unit_expr(other)], operator.and_))

    def __or__(self, other: Expr) -> Expr:
        return Expr(_uexp.NArgExpr([self._op, _to_unit_expr(other)], operator.or_))

    def __invert__(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(operator.inv)))

    def __xor__(self, other: Expr) -> Expr:
        return Expr(_uexp.NArgExpr([self._op, _to_unit_expr(other)], operator.xor))

    def add(self, other: Any) -> Expr:
        return Expr(_uexp.NArgExpr([self._op, _to_unit_expr(other)], operator.add))

    def sub(self, other: Any) -> Expr:
        return Expr(_uexp.NArgExpr([self._op, _to_unit_expr(other)], operator.sub))

    def mul(self, other: Any) -> Expr:
        return Expr(_uexp.NArgExpr([self._op, _to_unit_expr(other)], operator.mul))

    def truediv(self, other: Any) -> Expr:
        return Expr(_uexp.NArgExpr([self._op, _to_unit_expr(other)], operator.truediv))

    def pow(self, other: Any) -> Expr:
        return Expr(_uexp.NArgExpr([self._op, _to_unit_expr(other)], operator.pow))

    def eq(self, other: Any) -> Expr:
        return Expr(_uexp.NArgExpr([self._op, _to_unit_expr(other)], operator.eq))

    def ne(self, other: Any) -> Expr:
        return Expr(_uexp.NArgExpr([self._op, _to_unit_expr(other)], operator.ne))

    def lt(self, other: Any) -> Expr:
        return Expr(_uexp.NArgExpr([self._op, _to_unit_expr(other)], operator.lt))

    def le(self, other: Any) -> Expr:
        return Expr(_uexp.NArgExpr([self._op, _to_unit_expr(other)], operator.le))

    def gt(self, other: Any) -> Expr:
        return Expr(_uexp.NArgExpr([self._op, _to_unit_expr(other)], operator.gt))

    def ge(self, other: Any) -> Expr:
        return Expr(_uexp.NArgExpr([self._op, _to_unit_expr(other)], operator.ge))

    __add__ = add
    __sub__ = sub
    __mul__ = mul
    __truediv__ = truediv
    __pow__ = pow
    __eq__ = eq
    __ne__ = ne
    __lt__ = lt
    __le__ = le
    __gt__ = gt
    __ge__ = ge

    def unique(self, axis=None) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.unique, axis=axis)))

    def sin(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.sin)))

    def cos(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.cos)))

    def tan(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.tan)))

    def arcsin(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.arcsin)))

    def arccos(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.arccos)))

    def arctan(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.arctan)))

    def sinh(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.sinh)))

    def cosh(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.cosh)))

    def tanh(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.tanh)))

    def arcsinh(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.arcsinh)))

    def arccosh(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.arccosh)))

    def arctanh(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.arctanh)))

    def exp(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.exp)))

    def log(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.log)))

    def log2(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.log2)))

    def log10(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.log10)))

    def log1p(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.log1p)))

    def expm1(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.expm1)))

    def square(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.square)))

    def cbrt(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.cbrt)))

    def reciprocal(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.reciprocal)))

    def negative(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.negative)))

    def absolute(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.absolute)))

    def sign(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.sign)))

    def rint(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.rint)))

    def fix(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.fix)))

    def abs(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.abs)))

    def sqrt(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.sqrt)))

    def ceil(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.ceil)))

    def floor(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.floor)))

    def round(self, decimals=0) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.round, decimals=decimals)))

    def clip(self, a_min, a_max) -> Expr:
        return Expr(
            self._op.compose(_uexp.UfuncExpr(np.clip, a_min=a_min, a_max=a_max))
        )

    ##### Aggregation methods ########################################################
    def min(self, axis: Literal[None, 0] = None) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr.from_axis(np.min, axis=axis)))

    def max(self, axis: Literal[None, 0] = None) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr.from_axis(np.max, axis=axis)))

    def sum(self, axis: Literal[None, 0] = None) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr.from_axis(np.sum, axis=axis)))

    def mean(self, axis: Literal[None, 0] = None) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr.from_axis(np.mean, axis=axis)))

    def std(self, axis: Literal[None, 0] = None, ddof: int = 0) -> Expr:
        return Expr(
            self._op.compose(_uexp.UfuncExpr.from_axis(np.std, axis=axis, ddof=ddof))
        )

    def var(self, axis: Literal[None, 0] = None, ddof: int = 0) -> Expr:
        return Expr(
            self._op.compose(_uexp.UfuncExpr.from_axis(np.var, axis=axis, ddof=ddof))
        )

    def median(self, axis: Literal[None, 0] = None) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr.from_axis(np.median, axis=axis)))

    def percentile(self, q, axis: Literal[None, 0] = None) -> Expr:
        return Expr(
            self._op.compose(_uexp.UfuncExpr.from_axis(np.percentile, q=q, axis=axis))
        )

    def quantile(self, q, axis: Literal[None, 0] = None) -> Expr:
        return Expr(
            self._op.compose(_uexp.UfuncExpr.from_axis(np.quantile, q=q, axis=axis))
        )

    def all(self, axis: Literal[None, 0] = None) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr.from_axis(np.all, axis=axis)))

    def any(self, axis: Literal[None, 0] = None) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr.from_axis(np.any, axis=axis)))

    def argmin(self, axis: Literal[None, 0] = None) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr.from_axis(np.argmin, axis=axis)))

    def argmax(self, axis: Literal[None, 0] = None) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr.from_axis(np.argmax, axis=axis)))

    def len(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(_size)))

    ##### the "isXX" methods ########################################################
    def isin(self, values) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.isin, values)))

    def isnan(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.isnan)))

    def isfinite(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.isfinite)))

    def isinf(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.isinf)))

    def isfinite(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.isfinite)))

    def isposinf(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.isposinf)))

    def isneginf(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.isneginf)))

    def isreal(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.isreal)))

    def iscomplex(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.iscomplex)))

    def shape(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(_shape)))

    def concat(
        self, columns: IntoExpr | Sequence[IntoExpr], *more_columns: IntoExpr
    ) -> Expr:
        from structured_array._normalize import into_expr_multi

        exprs = into_expr_multi(columns, *more_columns)
        return Expr(_uexp.NArgExpr([expr._op for expr in exprs], _concat))

    def apply(self, func, *args, **kwargs) -> Expr:
        return Expr(_uexp.UfuncExpr(func, *args, **kwargs))

    def __getitem__(self, key) -> Expr:
        return Expr(_uexp.UfuncExpr(operator.getitem, key))


def _concat(*arr):
    return np.concatenate(arr, axis=-1)


def _shape(arr: np.ndarray):
    return np.array(arr.shape, dtype=int)


def _size(arr: np.ndarray):
    return np.int64(arr.size)


def _to_unit_expr(value) -> _uexp.UnitExpr:
    if isinstance(value, Expr):
        return value._op
    return _uexp.LitExpr(value)
