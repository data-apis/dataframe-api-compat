from __future__ import annotations

from tests.utils import bool_series_1


def test_column_any(library: str, request) -> None:
    ser = bool_series_1(library, request)
    result = ser.any()
    assert result


def test_column_all(library: str, request) -> None:
    ser = bool_series_1(library, request)
    result = ser.all()
    assert not result
