from __future__ import annotations

from tests.utils import maybe_collect
from tests.utils import nan_dataframe_1
from tests.utils import null_dataframe_2


def test_fill_null_column(library: str) -> None:
    df = null_dataframe_2(library).collect()
    ser = df.get_column_by_name("a")
    result = df.assign(ser.fill_null(0).rename("result")).relax()
    result = maybe_collect(result)["result"]
    assert result[2] == 0.0
    assert result[1] != 0.0
    assert result[0] != 0.0


def test_fill_null_noop_column(library: str) -> None:
    df = nan_dataframe_1(library).collect()
    ser = df.get_column_by_name("a")
    result = df.assign(ser.fill_null(0).rename("result")).relax()
    result = maybe_collect(result)["result"]
    if library != "pandas-numpy":
        # nan should not have changed!
        assert result[2] != result[2]
    else:
        # nan was filled with 0
        assert result[2] == 0
    assert result[1] != 0.0
    assert result[0] != 0.0
