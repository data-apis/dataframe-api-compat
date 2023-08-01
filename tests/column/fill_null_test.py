import pytest

from tests.utils import nan_series_1, null_dataframe_2


def test_fill_null_column(
    library: str,
    request: pytest.FixtureRequest,
) -> None:
    df = null_dataframe_2(library, request).get_column_by_name("a")
    result = df.fill_null(0)
    # friggin' impossible to test this due to pandas inconsistencies
    # with handling nan and null
    if library == "polars":
        assert result.column[2] == 0.0
        assert result.column[2] == 0.0
    else:
        assert result.column[2] == 0.0
        assert result.column[2] == 0.0


def test_fill_null_noop_column(library: str) -> None:
    ser = nan_series_1(library)
    result = ser.fill_null(0)
    # nan should not have changed!
    assert result.column[2] != result.column[2]
