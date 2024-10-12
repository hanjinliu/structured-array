from __future__ import annotations
from typing import Sequence
from structured_array.expression import Expr, SelectExpr
from structured_array.types import IntoExpr


def into_expr(value: IntoExpr) -> Expr:
    if isinstance(value, str):
        return Expr(SelectExpr(value))
    elif isinstance(value, Expr):
        return value
    else:
        raise TypeError(f"Expected str or Expr, got {type(value)}")


def into_expr_multi(
    columns: IntoExpr | Sequence[IntoExpr],
    *more_columns: IntoExpr,
    **named_columns: IntoExpr,
) -> list[Expr]:
    if isinstance(columns, (str, Expr)):
        columns = [columns]
    all_columns = [*columns, *more_columns]
    named = [into_expr(col).alias(name) for name, col in named_columns.items()]
    return [into_expr(col) for col in all_columns] + named
