from __future__ import annotations

from tests.utils import integer_dataframe_1


def test_column_len(library: str) -> None:
    result = len(integer_dataframe_1(library).col("a").persist())
    assert result == 3
