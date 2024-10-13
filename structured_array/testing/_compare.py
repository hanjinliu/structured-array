from __future__ import annotations

from numpy import testing as np_testing
import structured_array as st


def assert_array_equal(a: st.StructuredArray, b: st.StructuredArray):
    assert isinstance(a, st.StructuredArray)
    assert isinstance(b, st.StructuredArray)
    if a.columns != b.columns:  # pragma: no cover
        raise AssertionError(f"Columns do not match.\na: {a.columns}\nb: {b.columns}")
    if a.dtypes != b.dtypes:  # pragma: no cover
        raise AssertionError(f"Data types do not match.\na: {a.dtypes}\nb: {b.dtypes}")
    np_testing.assert_array_equal(a._arr, b._arr)
