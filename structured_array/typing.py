from typing import Union, TYPE_CHECKING, SupportsIndex

if TYPE_CHECKING:
    from structured_array.expression import Expr

IntoExpr = Union[str, "Expr"]
IntoIndex = Union[SupportsIndex, str]
