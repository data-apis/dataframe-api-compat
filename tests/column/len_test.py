from __future__ import annotations

from tests.utils import integer_series_1


def test_column_len(library: str) -> None:
    result = integer_series_1(library).len()
    assert result == 3
