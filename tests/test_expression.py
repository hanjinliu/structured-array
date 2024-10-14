import pytest
import structured_array as st
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose


def test_select():
    df = st.array({"col-0": [1, 2, 3], "b": [4, 5, 6]})
    st.testing.assert_array_equal(df.select("col-0"), st.array({"col-0": [1, 2, 3]}))
    st.testing.assert_array_equal(df.select("b"), st.array({"b": [4, 5, 6]}))
    st.testing.assert_array_equal(df.select("col-0", "b"), df)
    st.testing.assert_array_equal(
        df.select(st.col("col-0") + 1), st.array({"col-0": [2, 3, 4]})
    )
    st.testing.assert_array_equal(
        df.select(st.col("b") * 2), st.array({"b": [8, 10, 12]})
    )


def test_select_2d():
    df = st.array({"col-0": [[1, 2], [2, 3], [3, 4]], "b": [4, 5, 6]})
    assert_array_equal(df.select("col-0")["col-0"], [[1, 2], [2, 3], [3, 4]])


def test_alias():
    df = st.array({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert df.select(st.col("a").alias("c")).columns == ("c",)


def test_cast():
    df = st.array({"a": [1, 2, 3], "b": [4, 5, 6]})
    df.select(st.col("a").cast("float32")).schema["a"] == np.float32
    df.select(st.col("a").cast(np.float64)).schema["a"] == np.float64


def test_with_columns():
    df = st.array({"a": [1, 2, 3], "b": [4, 5, 6]})
    st.testing.assert_array_equal(
        df.with_columns(st.col("a").add(1).alias("c")),
        st.array({"a": [1, 2, 3], "b": [4, 5, 6], "c": [2, 3, 4]}),
    )


def test_with_columns_update():
    df = st.array({"a": [1, 2, 3], "b": [4, 5, 6]})
    st.testing.assert_array_equal(
        df.with_columns(st.col("a").add(1)),
        st.array({"a": [2, 3, 4], "b": [4, 5, 6]}),
    )


def test_with_columns_2d():
    df = st.array({"a": [[1, 2], [2, 3], [3, 4]], "b": [4, 5, 6]})
    st.testing.assert_array_equal(
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
    st.testing.assert_array_equal(
        df.with_columns(c=st.col("a") + 1),
        st.array({"a": [1, 2, 3], "b": [4, 5, 6], "c": [2, 3, 4]}),
    )


def test_with_columns_from_multiple():
    df = st.array({"a": [1, 2, 3], "b": [4, 5, 6]})
    st.testing.assert_array_equal(
        df.with_columns((st.col("a") + st.col("b")).alias("c")),
        st.array({"a": [1, 2, 3], "b": [4, 5, 6], "c": [5, 7, 9]}),
    )


def test_misc_methods():
    df = st.array({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert df.select(st.col("a").mean())[0, 0] == pytest.approx(2.0)
    assert df.select(st.col("a").std(ddof=0))[0, 0] == pytest.approx(
        np.std([1, 2, 3], ddof=0)
    )
    assert df.select(st.col("a").var(ddof=0))[0, 0] == pytest.approx(
        np.var([1, 2, 3], ddof=0)
    )
    assert df.select(st.col("a").sum())[0, 0] == 6
    assert df.select(st.col("a").min())[0, 0] == 1
    assert df.select(st.col("a").max())[0, 0] == 3
    assert df.select(st.col("a").len())[0, 0] == 3
    assert df.select(st.col("a").median())[0, 0] == 2
    assert df.select(st.col("a").percentile(50))[0, 0] == 2
    assert df.select(st.col("a").quantile(0.5))[0, 0] == 2
    assert df.select(st.col("a").lt(10).all())[0, 0] == True
    assert df.select(st.col("a").lt(2).any())[0, 0] == True
    assert df.select(st.col("a").argmin())[0, 0] == 0
    assert df.select(st.col("a").argmax())[0, 0] == 2
    assert df.select(st.col("a").ceil()).to_dict(asarray=False) == {"a": [1, 2, 3]}
    assert df.select(st.col("a").floor()).to_dict(asarray=False) == {"a": [1, 2, 3]}
    assert df.select(st.col("a").round()).to_dict(asarray=False) == {"a": [1, 2, 3]}
    assert df.select(st.col("a").clip(2, 3)).to_dict(asarray=False) == {"a": [2, 2, 3]}
    assert df.select(st.col("a").is_in([1, 3])).to_dict(asarray=False) == {
        "a": [True, False, True]
    }


def test_opertors():
    df = st.array({"a": [True, False, True, False], "b": [True, True, False, False]})
    assert df.select(st.col("a").not_())["a"].tolist() == [False, True, False, True]
    assert df.select(st.col("a").and_(st.col("b")))["a"].tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert df.select(st.col("a").or_(st.col("b")))["a"].tolist() == [
        True,
        True,
        True,
        False,
    ]
    assert df.select(st.col("a").xor(st.col("b")))["a"].tolist() == [
        False,
        True,
        True,
        False,
    ]

    df = st.array({"a": [1, 2, 3]})
    assert df.select(+st.col("a"))["a"].tolist() == [1, 2, 3]
    assert df.select(-st.col("a"))["a"].tolist() == [-1, -2, -3]
    assert df.select(st.col("a").truediv(2))["a"].tolist() == pytest.approx(
        [0.5, 1.0, 1.5]
    )
    assert df.select(st.col("a").floordiv(2))["a"].tolist() == [0, 1, 1]
    assert df.select(st.col("a").mod(2))["a"].tolist() == [1, 0, 1]
    assert df.select(st.col("a").pow(2))["a"].tolist() == [1, 4, 9]

    assert df.select(st.col("a").lt(2))["a"].tolist() == [True, False, False]
    assert df.select(st.col("a").le(2))["a"].tolist() == [True, True, False]
    assert df.select(st.col("a").gt(2))["a"].tolist() == [False, False, True]
    assert df.select(st.col("a").ge(2))["a"].tolist() == [False, True, True]
    assert df.select(st.col("a").eq(2))["a"].tolist() == [False, True, False]
    assert df.select(st.col("a").ne(2))["a"].tolist() == [True, False, True]
    assert df.select(st.col("a").is_between(1, 2))["a"].tolist() == [True, True, False]
    assert df.select(st.col("a").is_between(1, 2, closed="left"))["a"].tolist() == [
        True,
        False,
        False,
    ]
    assert df.select(st.col("a").is_between(1, 2, closed="right"))["a"].tolist() == [
        False,
        True,
        False,
    ]


def test_math_functions():
    val = np.array([0.3, 0.6, 0.9])
    df = st.array({"a": val})
    assert_allclose(df.select(st.col("a").abs())["a"], val)
    assert_allclose(df.select(st.col("a").exp())["a"], np.exp(val))
    assert_allclose(df.select(st.col("a").log())["a"], np.log(val))
    assert_allclose(df.select(st.col("a").log2())["a"], np.log2(val))
    assert_allclose(df.select(st.col("a").log10())["a"], np.log10(val))
    assert_allclose(df.select(st.col("a").sqrt())["a"], np.sqrt(val))
    assert_allclose(df.select(st.col("a").square())["a"], np.square(val))
    assert_allclose(df.select(st.col("a").cbrt())["a"], np.cbrt(val))
    assert_allclose(df.select(st.col("a").reciprocal())["a"], np.reciprocal(val))
    assert_allclose(df.select(st.col("a").sin())["a"], np.sin(val))
    assert_allclose(df.select(st.col("a").cos())["a"], np.cos(val))
    assert_allclose(df.select(st.col("a").tan())["a"], np.tan(val))
    assert_allclose(df.select(st.col("a").arcsin())["a"], np.arcsin(val))
    assert_allclose(df.select(st.col("a").arccos())["a"], np.arccos(val))
    assert_allclose(df.select(st.col("a").arctan())["a"], np.arctan(val))
    assert_allclose(df.select(st.col("a").sinh())["a"], np.sinh(val))
    assert_allclose(df.select(st.col("a").cosh())["a"], np.cosh(val))
    assert_allclose(df.select(st.col("a").tanh())["a"], np.tanh(val))
    assert_allclose(df.select(st.col("a").arcsinh())["a"], np.arcsinh(val))
    assert_allclose(df.select(st.col("a").add(2).arccosh())["a"], np.arccosh(val + 2))
    assert_allclose(df.select(st.col("a").arctanh())["a"], np.arctanh(val))
    assert_allclose(df.select(st.col("a").log1p())["a"], np.log1p(val))
    assert_allclose(df.select(st.col("a").rint())["a"], np.rint(val))
    assert_allclose(df.select(st.col("a").fix())["a"], np.fix(val))
    assert_allclose(df.select(st.col("a").sign())["a"], np.sign(val))


def test_degrees():
    df = st.array({"a": [0, np.pi / 2, np.pi], "b": [0, 45, 90]})
    assert_allclose(df.select(st.col("a").degrees())["a"], [0, 90, 180])
    assert_allclose(df.select(st.col("b").radians())["b"], [0, np.pi / 4, np.pi / 2])


def test_unique():
    df = st.array(
        {"a": [1, 2, 3, 1, 2, 3], "b": [[4, 1], [5, 4], [6, 3], [4, 1], [5, 4], [6, 2]]}
    )
    assert df.select(st.col("a").unique()).to_dict(asarray=False) == {"a": [1, 2, 3]}
    assert df.select(st.col("b").unique(axis=0))["b"].shape == (4, 2)


def test_concat():
    df = st.array({"a": [[1, 1], [2, 2]], "b": [[3, 3], [4, 4]]})
    assert df.select(st.col("a").concat(st.col("b")))["a"].tolist() == [
        [1, 1, 3, 3],
        [2, 2, 4, 4],
    ]


def test_shape():
    df = st.array({"a": [[1, 2, 3], [2, 2, 2]]})
    assert df.select(st.col("a").shape())[0, 0].tolist() == [2, 3]


def test_is_xx_methods():
    val = [np.inf, -np.inf, 0, 1, -1, np.nan]
    assert_array_equal(
        st.array({"a": val}).select(st.col("a").is_inf())["a"], np.isinf(val)
    )
    assert_array_equal(
        st.array({"a": val}).select(st.col("a").is_posinf())["a"], np.isposinf(val)
    )
    assert_array_equal(
        st.array({"a": val}).select(st.col("a").is_neginf())["a"], np.isneginf(val)
    )
    assert_array_equal(
        st.array({"a": val}).select(st.col("a").is_finite())["a"], np.isfinite(val)
    )
    assert_array_equal(
        st.array({"a": val}).select(st.col("a").is_nan())["a"], np.isnan(val)
    )

    val = [1, 1j, 1.0, 1.0j, 1.0 + 1.0j]
    assert_array_equal(
        st.array({"a": val}).select(st.col("a").is_real())["a"], np.isreal(val)
    )
    assert_array_equal(
        st.array({"a": val}).select(st.col("a").is_complex())["a"], np.iscomplex(val)
    )


def test_namespace_arr():
    ar = [[1, 2, 3], [2, 4, 6], [3, 2, 1]]
    df = st.array({"a": ar})
    assert_array_equal(df.select(st.col("a").arr.min())["a"], [1, 2, 1])
    assert_array_equal(df.select(st.col("a").arr.max())["a"], [3, 6, 3])
    assert_array_equal(df.select(st.col("a").arr.sum())["a"], [6, 12, 6])
    assert_array_equal(df.select(st.col("a").arr.mean())["a"], [2, 4, 2])
    assert_array_equal(
        df.select(st.col("a").arr.std(ddof=0))["a"], np.std(ar, axis=1, ddof=0)
    )
    assert_array_equal(
        df.select(st.col("a").arr.var(ddof=0))["a"], np.var(ar, axis=1, ddof=0)
    )
    assert_array_equal(df.select(st.col("a").arr.median())["a"], [2, 4, 2])
    assert_array_equal(df.select(st.col("a").arr.percentile(50))["a"], [2, 4, 2])
    assert_array_equal(df.select(st.col("a").arr.quantile(0.5))["a"], [2, 4, 2])
    assert_array_equal(
        df.select(st.col("a").lt(4.2).arr.all())["a"], [True, False, True]
    )
    assert_array_equal(
        df.select(st.col("a").lt(4.2).arr.any())["a"], [True, True, True]
    )
    assert_array_equal(df.select(st.col("a").arr.argmin())["a"], [0, 0, 2])
    assert_array_equal(df.select(st.col("a").arr.argmax())["a"], [2, 2, 0])
    assert_array_equal(df.select(st.col("a").arr[0])["a"], [1, 2, 3])


def test_namespace_arr_2d():
    ar = [
        [[1, 2, 3], [0, 0, 0]],
        [[2, 4, 6], [1, 1, 1]],
        [[3, 2, 1], [4, 4, 4]],
    ]
    df = st.array({"a": ar})
    assert_array_equal(df.select(st.col("a").arr.min(axis=0))["a"], np.min(ar, axis=1))
    assert_array_equal(df.select(st.col("a").arr.min(axis=1))["a"], np.min(ar, axis=2))
    assert_array_equal(
        df.select(st.col("a").arr.min(axis=(0,)))["a"], np.min(ar, axis=1)
    )

    assert_array_equal(
        df.select(st.col("a").arr.argmin(axis=0))["a"], np.argmin(ar, axis=1)
    )
    assert_array_equal(
        df.select(st.col("a").arr.argmin(axis=1))["a"], np.argmin(ar, axis=2)
    )
    assert_array_equal(df.select(st.col("a").arr[0, 1])["a"], [2, 4, 2])


def test_namespace_str():
    df = st.array({"a": ["ab", "bc", "Ab", "bAB"]})
    assert_array_equal(
        df.select(st.col("a").str.lower())["a"], ["ab", "bc", "ab", "bab"]
    )
    assert_array_equal(
        df.select(st.col("a").str.upper())["a"], ["AB", "BC", "AB", "BAB"]
    )
    assert_array_equal(
        df.select(st.col("a").str.title())["a"], ["Ab", "Bc", "Ab", "Bab"]
    )
    assert_array_equal(
        df.select(st.col("a").str.capitalize())["a"], ["Ab", "Bc", "Ab", "Bab"]
    )
    assert_array_equal(
        df.select(st.col("a").str.swapcase())["a"], ["AB", "BC", "aB", "Bab"]
    )
    assert_array_equal(df.select(st.col("a").str.len())["a"], [2, 2, 2, 3])
    assert_array_equal(
        df.select(st.col("a").str.center(5))["a"], ["  ab ", "  bc ", "  Ab ", " bAB "]
    )
    assert_array_equal(
        df.select(st.col("a").str.ljust(5))["a"], ["ab   ", "bc   ", "Ab   ", "bAB  "]
    )
    assert_array_equal(
        df.select(st.col("a").str.rjust(5))["a"], ["   ab", "   bc", "   Ab", "  bAB"]
    )
    assert_array_equal(
        df.select(st.col("a").str.zfill(5))["a"], ["000ab", "000bc", "000Ab", "00bAB"]
    )
    assert_array_equal(
        df.select(st.col("a").str.replace("b", "X"))["a"], ["aX", "Xc", "AX", "XAB"]
    )
    assert_array_equal(
        df.select(st.col("a").str.encode())["a"], [b"ab", b"bc", b"Ab", b"bAB"]
    )
    assert_array_equal(
        df.select(st.col("a").str.encode().str.decode())["a"], ["ab", "bc", "Ab", "bAB"]
    )

    # strip
    df = st.array({"a": [" ab", "bc ", "A b", "bAB"]})
    assert_array_equal(
        df.select(st.col("a").str.strip())["a"], ["ab", "bc", "A b", "bAB"]
    )
    assert_array_equal(
        df.select(st.col("a").str.lstrip())["a"], ["ab", "bc ", "A b", "bAB"]
    )
    assert_array_equal(
        df.select(st.col("a").str.rstrip())["a"], [" ab", "bc", "A b", "bAB"]
    )
    assert_array_equal(
        df.select(st.col("a").str.strip(" b"))["a"], ["a", "c", "A", "AB"]
    )

    # mod
    df = st.array({"a": ["a %s", "b %s", "c %s", "d %s"]})
    assert_array_equal(
        df.select(st.col("a").str.mod("X"))["a"], ["a X", "b X", "c X", "d X"]
    )

    # count
    df = st.array({"a": ["ab", "aab", "abb", "bbb"]})
    assert_array_equal(df.select(st.col("a").str.count("b"))["a"], [1, 1, 2, 3])

    # find
    assert_array_equal(df.select(st.col("a").str.find("b"))["a"], [1, 2, 1, 0])
    assert_array_equal(df.select(st.col("a").str.rfind("b"))["a"], [1, 2, 2, 2])

    # is_xx
    val = ["ab", "1", "a1", "1.0", "1.0a"]
    df = st.array({"a": val})
    assert_array_equal(df.select(st.col("a").str.is_alnum())["a"], np.char.isalnum(val))
    assert_array_equal(df.select(st.col("a").str.is_alpha())["a"], np.char.isalpha(val))
    assert_array_equal(df.select(st.col("a").str.is_digit())["a"], np.char.isdigit(val))
    assert_array_equal(df.select(st.col("a").str.is_lower())["a"], np.char.islower(val))
    assert_array_equal(df.select(st.col("a").str.is_upper())["a"], np.char.isupper(val))
    assert_array_equal(
        df.select(st.col("a").str.is_numeric())["a"], np.char.isnumeric(val)
    )
    assert_array_equal(df.select(st.col("a").str.is_space())["a"], np.char.isspace(val))
    assert_array_equal(df.select(st.col("a").str.is_title())["a"], np.char.istitle(val))
    assert_array_equal(
        df.select(st.col("a").str.is_decimal())["a"], np.char.isdecimal(val)
    )
