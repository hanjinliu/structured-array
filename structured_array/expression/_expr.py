from __future__ import annotations
import operator

import numpy as np
from structured_array.expression import _unitexpr as _uexp


class Expr:
    def __init__(self, op: _uexp.UnitExpr) -> None:
        self._op = op

    def _apply_expr(self, arr: np.ndarray) -> np.ndarray:
        return self._op.apply(arr)

    def alias(self, alias: str) -> Expr:
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

    def add(self, other: Expr) -> Expr:
        return Expr(_uexp.NArgExpr([self._op, _to_unit_expr(other)], operator.add))

    def sub(self, other: Expr) -> Expr:
        return Expr(_uexp.NArgExpr([self._op, _to_unit_expr(other)], operator.sub))

    def mul(self, other: Expr) -> Expr:
        return Expr(_uexp.NArgExpr([self._op, _to_unit_expr(other)], operator.mul))

    def truediv(self, other: Expr) -> Expr:
        return Expr(_uexp.NArgExpr([self._op, _to_unit_expr(other)], operator.truediv))

    def pow(self, other: Expr) -> Expr:
        return Expr(_uexp.NArgExpr([self._op, _to_unit_expr(other)], operator.pow))

    def eq(self, other: Expr) -> Expr:
        return Expr(_uexp.NArgExpr([self._op, _to_unit_expr(other)], operator.eq))

    def ne(self, other: Expr) -> Expr:
        return Expr(_uexp.NArgExpr([self._op, _to_unit_expr(other)], operator.ne))

    def lt(self, other: Expr) -> Expr:
        return Expr(_uexp.NArgExpr([self._op, _to_unit_expr(other)], operator.lt))

    def le(self, other: Expr) -> Expr:
        return Expr(_uexp.NArgExpr([self._op, _to_unit_expr(other)], operator.le))

    def gt(self, other: Expr) -> Expr:
        return Expr(_uexp.NArgExpr([self._op, _to_unit_expr(other)], operator.gt))

    def ge(self, other: Expr) -> Expr:
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

    def abs(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.abs)))

    def sqrt(self) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.sqrt)))

    def min(self, axis=None) -> Expr:
        return Expr(self._op.compose(_uexp.UfuncExpr(np.min, axis=axis)))


def _to_unit_expr(value) -> _uexp.UnitExpr:
    if isinstance(value, Expr):
        return value._op
    return _uexp.LitExpr(value)
