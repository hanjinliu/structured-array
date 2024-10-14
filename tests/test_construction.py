from pathlib import Path
import numpy as np
from numpy.testing import assert_array_equal
import structured_array as st
import pytest


@pytest.mark.parametrize(
    "d",
    [
        {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]},
        {"a": [1, 2, 3], "b": ["a", "b", "c"]},
    ],
)
def test_dict_of_1D_arrays(d):
    arr = st.array(d)
    assert arr.to_dict(asarray=False) == pytest.approx(d)


def test_dict_of_2D_arrays():
    arr = st.array(
        {"a": [[1, 2], [3, 4], [5, 6]], "b": [[3], [2], [1]], "c": [0, 1, 0]}
    )
    assert arr["a"].shape == (3, 2)
    assert arr["a"].tolist() == [[1, 2], [3, 4], [5, 6]]
    assert arr["b"].shape == (3, 1)
    assert arr["b"].tolist() == [[3], [2], [1]]
    assert arr["c"].tolist() == [0, 1, 0]


def test_dict_of_3D_arrays():
    arr = st.array(
        {
            "a": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            "b": [[[3], [2]], [[1], [0]]],
            "c": [[0, 1], [1, 0]],
        }
    )
    assert arr["a"].shape == (2, 2, 2)
    assert arr["a"].tolist() == [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    assert arr["b"].shape == (2, 2, 1)
    assert arr["b"].tolist() == [[[3], [2]], [[1], [0]]]
    assert arr["c"].tolist() == [[0, 1], [1, 0]]


def test_array_input_1d():
    arr = st.array(np.arange(10))
    assert arr.to_dict(asarray=False) == {"column_0": list(range(10))}


def test_array_input_2d():
    arr = st.array(np.arange(10).reshape(5, 2))
    assert arr.to_dict(asarray=False) == {
        "column_0": [0, 2, 4, 6, 8],
        "column_1": [1, 3, 5, 7, 9],
    }


def test_array_input_3d():
    arr = st.array(np.arange(30).reshape(5, 2, 3))
    assert arr.to_dict(asarray=False) == {
        "column_0": [[0, 1, 2], [6, 7, 8], [12, 13, 14], [18, 19, 20], [24, 25, 26]],
        "column_1": [[3, 4, 5], [9, 10, 11], [15, 16, 17], [21, 22, 23], [27, 28, 29]],
    }


def test_array_input_with_schema():
    ar = np.arange(6).reshape(3, 2)
    arr = st.array(ar, schema=["a", "bb"])
    assert arr.to_dict(asarray=False) == {"a": [0, 2, 4], "bb": [1, 3, 5]}
    arr = st.array(ar.tolist(), schema=["a", "bb"])
    assert arr.to_dict(asarray=False) == {"a": [0, 2, 4], "bb": [1, 3, 5]}


def test_empty_input():
    arr = st.array({})
    assert arr.to_dict(asarray=False) == {}


def test_schema():
    d = {"a": [1, 2, 3], "b": [4, 5, 6]}
    arr = st.array(d, schema={"a": np.uint16, "b": np.float32})
    assert arr.schema == {"a": np.uint16, "b": np.float32}
    arr = st.array(d, schema=[("a", "uint16"), ("b", "float32")])
    assert arr.schema == {"a": np.uint16, "b": np.float32}
    with pytest.raises(ValueError):
        st.array(d, schema={"a": np.uint16, "c": np.float32})


def test_pandas():
    import pandas as pd

    d = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}
    df = pd.DataFrame(d)
    arr = st.array(df)
    assert arr.to_dict(asarray=False) == pytest.approx(d)
    st.array(df, schema={"a": np.uint16, "b": np.float32})
    with pytest.raises(ValueError):
        st.array(df, schema={"a": np.uint16, "c": np.float32})


def test_polars():
    import polars as pl

    d = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}
    df = pl.DataFrame(d)
    arr = st.array(df)
    assert arr.to_dict(asarray=False) == pytest.approx(d)
    st.array(df, schema={"a": np.uint16, "b": np.float32})
    with pytest.raises(ValueError):
        st.array(df, schema={"a": np.uint16, "c": np.float32})


def test_io(tmp_path: Path):
    ar = st.array({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    path = tmp_path / "test.npy"
    ar.write_npy(path)
    ar2 = st.read_npy(path)
    assert_array_equal(ar, ar2)


@pytest.mark.parametrize(
    "d",
    [
        {"a": [1, 2, 3], "b": [4.0, 5.0]},
        {"a": [1, 2, 3], "b": ["a", "b", "c", "d"]},
    ],
)
def test_bad_input(d):
    with pytest.raises(ValueError):
        st.array(d)
