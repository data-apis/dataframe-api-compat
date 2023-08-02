from __future__ import annotations

from tests.utils import integer_series_1


def test_get_value(library: str, request) -> None:
    result = integer_series_1(library, request).get_value(0)
    assert result == 1
