import structured_array as st


def test_groupby_iteration():
    df = st.array({"a": [1, 2, 1, 2], "b": [4, 5, 6, 7]})
    groups = list(df.group_by("a"))
    assert len(groups) == 2
    assert list(groups[0][0]) == [1]
    assert list(groups[1][0]) == [2]
    assert groups[0][1].to_dict(asarray=False) == {"a": [1, 1], "b": [4, 6]}
    assert groups[1][1].to_dict(asarray=False) == {"a": [2, 2], "b": [5, 7]}


def test_agg():
    df = st.array({"a": [1, 2, 1, 2], "b": [4, 5, 6, 7]})
    assert df.group_by("a").agg(st.col("b").sum()).to_dict(asarray=False) == {
        "a": [1, 2],
        "b": [10, 12],
    }
    assert df.group_by("a").agg(st.col("b").sum().alias("total")).to_dict(
        asarray=False
    ) == {"a": [1, 2], "total": [10, 12]}
    assert df.group_by("a").agg(st.col("b").min().alias("min")).to_dict(
        asarray=False
    ) == {"a": [1, 2], "min": [4, 5]}
    assert df.group_by("a").agg(st.col("b").max().alias("max")).to_dict(
        asarray=False
    ) == {"a": [1, 2], "max": [6, 7]}
