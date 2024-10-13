from typing import Union, TYPE_CHECKING, SupportsIndex
import numpy as np

if TYPE_CHECKING:
    from structured_array.expression import Expr

IntoExpr = Union[str, "Expr"]
IntoIndex = Union[SupportsIndex, str]
IntoDType = tuple[str, np.dtype, tuple[int, ...]]
