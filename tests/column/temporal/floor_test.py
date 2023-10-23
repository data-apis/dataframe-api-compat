from __future__ import annotations

from datetime import datetime

import pytest

from tests.utils import temporal_dataframe_1


@pytest.mark.parametrize(
    ("freq", "expected"),
    [
        ("1day", [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)]),
    ],
)
def test_floor(library: str, freq: str, expected: list[datetime]) -> None:
    df = temporal_dataframe_1(library)
    col = df.__dataframe_namespace__().col
    result = df.select(col("a").dt.floor(freq)).collect().col("a")
    assert result.get_value(0) == expected[0]
    assert result.get_value(1) == expected[1]
    assert result.get_value(2) == expected[2]
