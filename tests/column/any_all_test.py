from __future__ import annotations

from tests.utils import bool_series_1


def test_column_any(library: str) -> None:
    ser = bool_series_1(library)
    result = ser.any()
    assert result


def test_column_all(library: str) -> None:
    ser = bool_series_1(library)
    result = ser.all()
    assert not result
