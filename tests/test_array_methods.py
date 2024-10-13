import numpy as np
import structured_array as st


def test_head_tail():
    df = st.array({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert df.head(2).to_dict(asarray=False) == {"a": [1, 2], "b": [4, 5]}
    assert df.head(10).to_dict(asarray=False) == {"a": [1, 2, 3], "b": [4, 5, 6]}
    assert df.tail(2).to_dict(asarray=False) == {"a": [2, 3], "b": [5, 6]}
    assert df.tail(10).to_dict(asarray=False) == {"a": [1, 2, 3], "b": [4, 5, 6]}


def test_iteration():
    df = st.array({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert [list(a) for a in df.iter_rows()] == [[1, 4], [2, 5], [3, 6]]
    assert [list(a) for a in df.iter_columns()] == [[1, 2, 3], [4, 5, 6]]


def test_filter():
    df = st.array({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert df.filter(st.col("a") > 1).to_dict(asarray=False) == {
        "a": [2, 3],
        "b": [5, 6],
    }


def test_sort():
    df = st.array({"a": [3, 2, 1], "b": [6, 4, 5]})
    assert df.sort("a").to_dict(asarray=False) == {"a": [1, 2, 3], "b": [5, 4, 6]}
    assert df.sort("a", ascending=False).to_dict(asarray=False) == {
        "a": [3, 2, 1],
        "b": [6, 4, 5],
    }
    assert df.sort(-st.col("a")).to_dict(asarray=False) == {
        "a": [3, 2, 1],
        "b": [6, 4, 5],
    }


def test_join():
    df0 = st.array({"a": [1, 2, 3], "b": [4, 5, 6]})
    df1 = st.array({"a": [1, 2, 3], "c": [7, 8, 9]})
    df = df0.join(df1, on="a")
    assert df.to_dict(asarray=False) == {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}


def test_repr():
    df = st.array(
        {"a": [1, 2, 3], "b": ["a", "bb", "ccc"], "c": [[1, 2], [3, 4], [5, 6]]}
    )
    repr(df)
    repr(st.array({"a": np.arange(30)}))


def test_getitem():
    df = st.array({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert list(df["a"]) == [1, 2, 3]
    assert df[1:3].to_dict(asarray=False) == {"a": [2, 3], "b": [5, 6]}
    assert df[1].to_dict(asarray=False) == {"a": [2], "b": [5]}
    assert df[1, "a"] == 2
    assert df[1, 1] == 5
    assert list(df[1, 1:2]) == [5]
    assert list(df[1:3, "a"]) == [2, 3]
    assert df[1:3, 1:2].to_dict(asarray=False) == {"b": [5, 6]}
