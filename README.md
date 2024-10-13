# structured-array

[![PyPI - Version](https://img.shields.io/pypi/v/structured-array.svg)](https://pypi.org/project/structured-array)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/structured-array.svg)](https://pypi.org/project/structured-array)

Efficient manipulation of the numpy structured arrays.

-----

## Table of Contents

- [structured-array](#structured-array)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Examples](#examples)
  - [License](#license)

## Installation

```console
pip install structured-array
```

## Examples

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

## License

`structured-array` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
