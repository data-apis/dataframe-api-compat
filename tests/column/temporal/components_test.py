from __future__ import annotations

import pytest

from tests.utils import temporal_dataframe_1


@pytest.mark.parametrize("attr", ["year", "month", "day", "hour", "minute", "second"])
def test_expr_components(library: str, attr: str) -> None:
    attr = "year"
    expected = [2020, 2020, 2020]
    df = temporal_dataframe_1(library)
    col = df.__dataframe_namespace__().col
    result = df.select(getattr(col("a").dt, attr)())
    result_list = result.collect().get_column_by_name("a")
    assert result_list.get_value(0) == expected[0]
    assert result_list.get_value(1) == expected[1]
    assert result_list.get_value(2) == expected[2]


@pytest.mark.parametrize("attr", ["year", "month", "day", "hour", "minute", "second"])
def test_col_components(library: str, attr: str) -> None:
    attr = "year"
    expected = [2020, 2020, 2020]
    df = temporal_dataframe_1(library).collect()
    result = getattr(df.get_column_by_name("a").dt, attr)()
    assert result.get_value(0) == expected[0]
    assert result.get_value(1) == expected[1]
    assert result.get_value(2) == expected[2]
