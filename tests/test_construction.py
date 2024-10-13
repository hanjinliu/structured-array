import numpy as np
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


def test_schema():
    arr = st.array(
        {"a": [1, 2, 3], "b": [4, 5, 6]}, schema={"a": np.uint16, "b": np.float32}
    )
    assert arr.schema == {"a": np.uint16, "b": np.float32}


def test_pandas():
    import pandas as pd

    d = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}
    df = pd.DataFrame(d)
    arr = st.array(df)
    assert arr.to_dict(asarray=False) == pytest.approx(d)


def test_polars():
    import polars as pl

    d = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}
    df = pl.DataFrame(d)
    arr = st.array(df)
    assert arr.to_dict(asarray=False) == pytest.approx(d)


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
