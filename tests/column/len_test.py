from __future__ import annotations

from typing import TYPE_CHECKING

from tests.utils import integer_series_1

if TYPE_CHECKING:
    import pytest


def test_column_len(library: str, request: pytest.FixtureRequest) -> None:
    result = len(integer_series_1(library, request))
    assert result == 3
