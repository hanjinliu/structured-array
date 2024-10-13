import structured_array as st
import numpy as np
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


def test_alias():
    df = st.array({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert df.select(st.col("a").alias("c")).columns == ("c",)


def test_cast():
    df = st.array({"a": [1, 2, 3], "b": [4, 5, 6]})
    df.select(st.col("a").cast("float32")).schema["a"] == np.float32
    df.select(st.col("a").cast(np.float64)).schema["a"] == np.float64


def test_with_columns():
    df = st.array({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert_array_equal(
        df.with_columns(st.col("a").add(1).alias("c")),
        st.array({"a": [1, 2, 3], "b": [4, 5, 6], "c": [2, 3, 4]}),
    )


def test_with_columns_update():
    df = st.array({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert_array_equal(
        df.with_columns(st.col("a").add(1)),
        st.array({"a": [2, 3, 4], "b": [4, 5, 6]}),
    )


def test_with_columns_2d():
    df = st.array({"a": [[1, 2], [2, 3], [3, 4]], "b": [4, 5, 6]})
    assert_array_equal(
        df.with_columns(st.col("a").sub(1).alias("c")),
        st.array(
            {
                "a": [[1, 2], [2, 3], [3, 4]],
                "b": [4, 5, 6],
                "c": [[0, 1], [1, 2], [2, 3]],
            }
        ),
    )


def test_with_columns_named():
    df = st.array({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert_array_equal(
        df.with_columns(c=st.col("a") + 1),
        st.array({"a": [1, 2, 3], "b": [4, 5, 6], "c": [2, 3, 4]}),
    )


def test_with_columns_from_multiple():
    df = st.array({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert_array_equal(
        df.with_columns((st.col("a") + st.col("b")).alias("c")),
        st.array({"a": [1, 2, 3], "b": [4, 5, 6], "c": [5, 7, 9]}),
    )
