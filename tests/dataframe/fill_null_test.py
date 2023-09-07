from __future__ import annotations

import pytest

from tests.utils import interchange_to_pandas
from tests.utils import maybe_collect
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
def test_fill_null(
    library: str,
    column_names: list[str] | None,
) -> None:
    df = null_dataframe_2(library)
    result = df.fill_null(0, column_names=column_names)

    if column_names is None or "a" in column_names:
        res1 = result.filter(result.get_column_by_name("a").is_null())
        res1 = maybe_collect(res1, library)
        # check there no nulls left in the column
        assert res1.__dataframe__().num_rows() == 0
        # check the last element was filled with 0
        assert interchange_to_pandas(result, library)["a"].iloc[2] == 0
    if column_names is None or "b" in column_names:
        res1 = result.filter(result.get_column_by_name("b").is_null())
        res1 = maybe_collect(res1, library)
        assert res1.__dataframe__().num_rows() == 0
        assert interchange_to_pandas(result, library)["b"].iloc[2] == 0


def test_fill_null_noop(library: str) -> None:
    df = nan_dataframe_1(library)
    result = df.fill_null(0)
    if hasattr(result.dataframe, "collect"):
        result = result.dataframe.collect()
    else:
        result = result.dataframe
    if library != "pandas-numpy":
        # nan should not have changed!
        assert result["a"][2] != result["a"][2]
    else:
        # in pandas-numpy, null is nan, so it gets filled
        assert result["a"][2] == 0
