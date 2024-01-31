from __future__ import annotations

import pytest

from tests.utils import nan_dataframe_1
from tests.utils import null_dataframe_2


@pytest.mark.parametrize(
    "column_names",
    [
        ["a", "b"],
        None,
        ["a"],
        ["b"],
    ],
)
def test_fill_null(library: str, column_names: list[str] | None) -> None:
    df = null_dataframe_2(library)
    df.__dataframe_namespace__()
    result = df.fill_null(0, column_names=column_names).persist()

    if column_names is None or "a" in column_names:
        res1 = result.filter(result.get_column("a").is_null())
        # check there no nulls left in the column
        assert res1.shape()[0] == 0
        # check the last element was filled with 0
        assert result.col("a").persist().get_value(2).scalar == 0
    if column_names is None or "b" in column_names:
        res1 = result.filter(result.get_column("b").is_null())
        assert res1.shape()[0] == 0
        assert result.col("b").persist().get_value(2).scalar == 0


def test_fill_null_noop(library: str) -> None:
    df = nan_dataframe_1(library)
    result_raw = df.fill_null(0)
    if hasattr(result_raw.dataframe, "collect"):
        result = result_raw.dataframe.collect()
    else:
        result = result_raw.dataframe
    if library != "pandas-numpy":
        # nan should not have changed!
        assert result["a"][2] != result["a"][2]
    else:
        # in pandas-numpy, null is nan, so it gets filled
        assert result["a"][2] == 0
