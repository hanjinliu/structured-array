from __future__ import annotations

from numpy import testing as np_testing
import structured_array as st


def assert_array_equal(a: st.StructuredArray, b: st.StructuredArray):
    if not isinstance(a, st.StructuredArray):
        a = st.array(a)
    if not isinstance(b, st.StructuredArray):
        b = st.array(b)
    if a.columns != b.columns:
        raise AssertionError(f"Columns do not match.\na: {a.columns}\nb: {b.columns}")
    if a.dtypes != b.dtypes:
        raise AssertionError(f"Data types do not match.\na: {a.dtypes}\nb: {b.dtypes}")
    np_testing.assert_array_equal(a._arr, b._arr)
