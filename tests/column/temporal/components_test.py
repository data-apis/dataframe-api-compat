from __future__ import annotations

import pytest

from tests.utils import temporal_dataframe_1


@pytest.mark.parametrize("attr", ["year", "month", "day", "hour", "minute", "second"])
def test_components(library: str, attr: str) -> None:
    attr = "year"
    expected = [2020, 2020, 2020]
    df = temporal_dataframe_1(library)
    col = df.__dataframe_namespace__().col
    result = df.select(getattr(col("a").dt, attr)())
    result_list = result.collect().get_column_by_name("a")
    assert result_list[0] == expected[0]
    assert result_list[1] == expected[1]
    assert result_list[2] == expected[2]
