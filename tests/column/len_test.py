from __future__ import annotations

from tests.utils import integer_series_1


def test_column_len(library: str, request) -> None:
    result = len(integer_series_1(library, request))
    assert result == 3
