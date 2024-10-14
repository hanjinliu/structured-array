from typing import Sequence, Union, TYPE_CHECKING, SupportsIndex, Mapping
import numpy as np

if TYPE_CHECKING:
    from structured_array.expression import Expr
    from numpy.typing import DTypeLike

IntoExpr = Union[str, "Expr"]
IntoIndex = Union[SupportsIndex, str]
IntoDType = tuple[str, np.dtype, tuple[int, ...]]
SchemaType = Union[Mapping[str, "DTypeLike"], Sequence[tuple[str, "DTypeLike"]]]
AxisType = Union[None, int, Sequence[int]]
