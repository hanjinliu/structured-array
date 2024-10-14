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
    repr(
        st.array(
            {
                "aaaaaaaaaaaaaaaaa": np.arange(-10, 10, dtype=np.int64) * 1e10,
                "b": np.zeros(20),
                "c": ["a" * i for i in range(20)],
                "d": np.zeros(20),
                "e": np.zeros(20),
                "f": np.zeros(20),
                "g": np.zeros(20),
                "h": np.zeros(20),
                "i": np.zeros(20),
                "12345678": np.zeros(20),
            }
        )
    )


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


def test_misc():
    df = st.array({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert "a" in df
    assert "c" not in df
    assert len(df) == 3
    assert df.shape == (3, 2)


def test_rename():
    df = st.array({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert df.rename({"a": "A"}).to_dict(asarray=False) == {
        "A": [1, 2, 3],
        "b": [4, 5, 6],
    }

    df = st.array({"a": [[1, 2], [2, 3], [3, 2]], "b": [4, 5, 6]})
    assert df.rename({"a": "A"}).to_dict(asarray=False) == {
        "A": [[1, 2], [2, 3], [3, 2]],
        "b": [4, 5, 6],
    }


def test_arange():
    df = st.array({"a": [0, 0, 0]})
    assert df.with_columns(st.arange()).to_dict(asarray=False) == {
        "a": [0, 0, 0],
        "arange": [0, 1, 2],
    }
    assert df.with_columns(b=st.arange(stop=10)).to_dict(asarray=False) == {
        "a": [0, 0, 0],
        "b": [7, 8, 9],
    }
    assert df.with_columns(st.arange(start=1)).to_dict(asarray=False) == {
        "a": [0, 0, 0],
        "arange": [1, 2, 3],
    }
    assert df.with_columns(st.arange(step=2)).to_dict(asarray=False) == {
        "a": [0, 0, 0],
        "arange": [0, 2, 4],
    }


def test_linspace():
    df = st.array({"a": [0, 0, 0]})
    assert df.with_columns(st.linspace(0, 1)).to_dict(asarray=False) == {
        "a": [0, 0, 0],
        "linspace": [0.0, 0.5, 1.0],
    }
    assert df.with_columns(st.linspace(0, 3, endpoint=False)).to_dict(
        asarray=False
    ) == {"a": [0, 0, 0], "linspace": [0.0, 1.0, 2.0]}
