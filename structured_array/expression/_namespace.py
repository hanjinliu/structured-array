from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Self
    from structured_array.expression._unitexpr import UnitExpr
    from structured_array.expression import Expr


class ExprNamespace:
    def __init__(self, expr: Expr | None = None) -> None:
        self._expr_or_none = expr

    def __get__(self, obj, objtype=None) -> Self:
        if obj is None:
            return self
        return self.__class__(obj)

    def _expr(self) -> Expr:
        if self._expr_or_none is None:
            raise AttributeError("Only Expr instances have this attribute")
        return self._expr_or_none

    def _op(self) -> UnitExpr:
        return self._expr()._op

    @classmethod
    def _new(cls, op: UnitExpr) -> Expr:
        from structured_array.expression import Expr

        return Expr(op)
