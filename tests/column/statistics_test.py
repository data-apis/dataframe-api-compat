from tests.utils import integer_series_5, integer_series_1


def test_mean(library: str) -> None:
    result = integer_series_5(library).mean()
    assert result == 2.0


def test_std(library: str) -> None:
    result = integer_series_5(library).std()
    assert abs(result - 1.7320508075688772) < 1e-8


def test_column_max(library: str) -> None:
    result = integer_series_1(library).max()
    assert result == 3
