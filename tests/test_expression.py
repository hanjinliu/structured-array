import structured_array as st
from numpy.testing import assert_array_equal


def test_select():
    df = st.array({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert_array_equal(df.select("a"), st.array({"a": [1, 2, 3]}))
    assert_array_equal(df.select("b"), st.array({"b": [4, 5, 6]}))
    assert_array_equal(df.select("a", "b"), df)
    assert_array_equal(df.select(st.col("a") + 1), st.array({"a": [2, 3, 4]}))


def test_select_2d():
    df = st.array({"a": [[1, 2], [2, 3], [3, 4]], "b": [4, 5, 6]})
    assert_array_equal(df.select("a")["a"], [[1, 2], [2, 3], [3, 4]])
