from __future__ import annotations

from tests.utils import integer_dataframe_1


def test_column_len(library: str) -> None:
    result = integer_dataframe_1(library).persist().get_column("a").n_unique().scalar
    assert result == 3
