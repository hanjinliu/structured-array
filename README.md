# structured-array

[![PyPI - Version](https://img.shields.io/pypi/v/structured-array.svg)](https://pypi.org/project/structured-array)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/structured-array.svg)](https://pypi.org/project/structured-array)
[![codecov](https://codecov.io/gh/hanjinliu/structured-array/graph/badge.svg?token=vaPM3dusOW)](https://codecov.io/gh/hanjinliu/structured-array)

Efficient manipulation of the numpy structured arrays.

-----

## Installation

```console
pip install structured-array
```

## Examples

A structured array can easily be created.

```python
import structured_array as st

arr = st.array({
    "label": ["a", "b", "c"],
    "value": [4, 5, 6],
    "array": [np.zeros(3), np.ones(3), np.zeros(3)],
})
arr
```

```
label    value    array
[<U1]    [<i8]    [<f8]
-------  -------  ----------
a        4        (3,) array
b        5        (3,) array
c        6        (3,) array
```

You can directly read and write the structured array from/to a npy file.

```python
arr = st.read_npy("data.npy")
```

`structured-array` use [polars expression system](https://docs.pola.rs/user-guide/concepts/expressions-and-contexts/)
to manipulate the structured array.

```python
arr.select("label", st.col("value") + 1)  # column selection
arr.group_by("label").agg(st.col("value").sum())  # aggregation
arr.filter(st.col("value") > 5)  # filtering
```

## License

`structured-array` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
