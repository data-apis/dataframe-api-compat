from __future__ import annotations

from typing import TYPE_CHECKING

from tests.utils import integer_series_1

if TYPE_CHECKING:
    import pytest


def test_rename(library: str, request: pytest.FixtureRequest) -> None:
    ser = integer_series_1(library, request)
    result = ser.rename("new_name")
    assert result.name == "new_name"
