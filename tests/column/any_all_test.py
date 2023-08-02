from __future__ import annotations

from typing import TYPE_CHECKING

from tests.utils import bool_series_1

if TYPE_CHECKING:
    import pytest


def test_column_any(library: str, request: pytest.FixtureRequest) -> None:
    ser = bool_series_1(library, request)
    result = ser.any()
    assert result


def test_column_all(library: str, request: pytest.FixtureRequest) -> None:
    ser = bool_series_1(library, request)
    result = ser.all()
    assert not result
