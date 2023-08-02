from __future__ import annotations

from typing import TYPE_CHECKING

from tests.utils import integer_series_1

if TYPE_CHECKING:
    import pytest


def test_get_value(library: str, request: pytest.FixtureRequest) -> None:
    result = integer_series_1(library, request).get_value(0)
    assert result == 1
