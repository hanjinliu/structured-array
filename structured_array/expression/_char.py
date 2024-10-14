from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from structured_array.expression._namespace import ExprNamespace
from structured_array.expression import _unitexpr as _uexp

if TYPE_CHECKING:
    from structured_array.expression import Expr


class StrNamespace(ExprNamespace):
    def lower(self) -> Expr:
        return self._new(self._op().compose(_uexp.UfuncExpr(np.char.lower)))

    def upper(self) -> Expr:
        return self._new(self._op().compose(_uexp.UfuncExpr(np.char.upper)))

    def title(self) -> Expr:
        return self._new(self._op().compose(_uexp.UfuncExpr(np.char.title)))

    def capitalize(self) -> Expr:
        return self._new(self._op().compose(_uexp.UfuncExpr(np.char.capitalize)))

    def swapcase(self) -> Expr:
        return self._new(self._op().compose(_uexp.UfuncExpr(np.char.swapcase)))

    def len(self) -> Expr:
        return self._new(self._op().compose(_uexp.UfuncExpr(np.char.str_len)))

    def strip(self, chars=None) -> Expr:
        return self._new(self._op().compose(_uexp.UfuncExpr(np.char.strip, chars)))

    def lstrip(self, chars=None) -> Expr:
        return self._new(self._op().compose(_uexp.UfuncExpr(np.char.lstrip, chars)))

    def rstrip(self, chars=None) -> Expr:
        return self._new(self._op().compose(_uexp.UfuncExpr(np.char.rstrip, chars)))

    def center(self, width, fillchar=" ") -> Expr:
        return self._new(
            self._op().compose(_uexp.UfuncExpr(np.char.center, width, fillchar))
        )

    def ljust(self, width, fillchar=" ") -> Expr:
        return self._new(
            self._op().compose(_uexp.UfuncExpr(np.char.ljust, width, fillchar))
        )

    def rjust(self, width, fillchar=" ") -> Expr:
        return self._new(
            self._op().compose(_uexp.UfuncExpr(np.char.rjust, width, fillchar))
        )

    def zfill(self, width) -> Expr:
        return self._new(self._op().compose(_uexp.UfuncExpr(np.char.zfill, width)))

    def replace(self, old, new, count=-1) -> Expr:
        return self._new(
            self._op().compose(_uexp.UfuncExpr(np.char.replace, old, new, count))
        )

    def decode(self, encoding: str | None = None, errors: str | None = None) -> Expr:
        return self._new(
            self._op().compose(_uexp.UfuncExpr(np.char.decode, encoding, errors))
        )

    def encode(self, encoding: str | None = None, errors: str | None = None) -> Expr:
        return self._new(
            self._op().compose(_uexp.UfuncExpr(np.char.encode, encoding, errors))
        )

    def mod(self, value: Any) -> Expr:
        return self._new(self._op().compose(_uexp.UfuncExpr(np.char.mod, value)))

    def count(self, sub: str, start: int = 0, end: int | None = None) -> Expr:
        return self._new(
            self._op().compose(_uexp.UfuncExpr(np.char.count, sub, start, end))
        )

    def find(self, sub: str, start: int = 0, end: int | None = None) -> Expr:
        return self._new(
            self._op().compose(_uexp.UfuncExpr(np.char.find, sub, start, end))
        )

    def rfind(self, sub: str, start: int = 0, end: int | None = None) -> Expr:
        return self._new(
            self._op().compose(_uexp.UfuncExpr(np.char.rfind, sub, start, end))
        )

    def startswith(self, prefix: str, start: int = 0, end: int | None = None) -> Expr:
        return self._new(
            self._op().compose(_uexp.UfuncExpr(np.char.startswith, prefix, start, end))
        )

    def endswith(self, suffix: str, start: int = 0, end: int | None = None) -> Expr:
        return self._new(
            self._op().compose(_uexp.UfuncExpr(np.char.endswith, suffix, start, end))
        )

    def is_alnum(self) -> Expr:
        """Return True if an element is alphabetic or numeric character."""
        return self._new(self._op().compose(_uexp.UfuncExpr(np.char.isalnum)))

    def is_alpha(self) -> Expr:
        """Return True if an element is alphabetic character."""
        return self._new(self._op().compose(_uexp.UfuncExpr(np.char.isalpha)))

    def is_digit(self) -> Expr:
        """Return True if an element is digit character."""
        return self._new(self._op().compose(_uexp.UfuncExpr(np.char.isdigit)))

    def is_numeric(self) -> Expr:
        """Return True if an element is numeric character."""
        return self._new(self._op().compose(_uexp.UfuncExpr(np.char.isnumeric)))

    def is_space(self) -> Expr:
        """Return True if an element is whitespace character."""
        return self._new(self._op().compose(_uexp.UfuncExpr(np.char.isspace)))

    def is_lower(self) -> Expr:
        """Return True if an element is lowercase character."""
        return self._new(self._op().compose(_uexp.UfuncExpr(np.char.islower)))

    def is_upper(self) -> Expr:
        """Return True if an element is uppercase character."""
        return self._new(self._op().compose(_uexp.UfuncExpr(np.char.isupper)))

    def is_title(self) -> Expr:
        """Return True if an element is titlecase character."""
        return self._new(self._op().compose(_uexp.UfuncExpr(np.char.istitle)))

    def is_decimal(self) -> Expr:
        """Return True if an element is decimal character."""
        return self._new(self._op().compose(_uexp.UfuncExpr(np.char.isdecimal)))
