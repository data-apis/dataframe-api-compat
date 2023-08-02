from __future__ import annotations

from tests.utils import integer_series_1
from tests.utils import integer_series_5


def test_mean(library: str, request) -> None:
    result = integer_series_5(library, request).mean()
    assert result == 2.0


def test_std(library: str, request) -> None:
    result = integer_series_5(library, request).std()
    assert abs(result - 1.7320508075688772) < 1e-8


def test_column_max(library: str, request) -> None:
    result = integer_series_1(library, request).max()
    assert result == 3
