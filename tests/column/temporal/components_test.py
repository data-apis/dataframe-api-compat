from __future__ import annotations

import pytest

from tests.utils import temporal_dataframe_1


@pytest.mark.parametrize(
    ("attr", "expected"),
    [
        ("year", [2020, 2020, 2020]),
        ("month", [1, 1, 1]),
        ("day", [1, 2, 3]),
        ("hour", [1, 3, 5]),
        ("minute", [2, 1, 4]),
        ("second", [1, 2, 9]),
        ("iso_weekday", [3, 4, 5]),
    ],
)
def test_expr_components(library: str, attr: str, expected: list[int]) -> None:
    df = temporal_dataframe_1(library)
    col = df.__dataframe_namespace__().col
    for col_name in ("a", "c", "e"):
        result = df.select(getattr(col(col_name).dt, attr)())
        result_list = result.collect().get_column_by_name(col_name)
        assert result_list.get_value(0) == expected[0]
        assert result_list.get_value(1) == expected[1]
        assert result_list.get_value(2) == expected[2]


@pytest.mark.parametrize(
    ("attr", "expected"),
    [
        ("year", [2020, 2020, 2020]),
        ("month", [1, 1, 1]),
        ("day", [1, 2, 3]),
        ("hour", [1, 3, 5]),
        ("minute", [2, 1, 4]),
        ("second", [1, 2, 9]),
        ("iso_weekday", [3, 4, 5]),
    ],
)
def test_col_components(library: str, attr: str, expected: list[int]) -> None:
    df = temporal_dataframe_1(library).collect()
    for col_name in ("a", "c", "e"):
        result = getattr(df.get_column_by_name(col_name).dt, attr)()
        assert result.get_value(0) == expected[0]
        assert result.get_value(1) == expected[1]
        assert result.get_value(2) == expected[2]


@pytest.mark.parametrize(
    ("col_name", "expected"),
    [
        ("a", [123000, 321000, 987000]),
        ("c", [123543, 321654, 987321]),
        ("e", [123543, 321654, 987321]),
    ],
)
def test_expr_microsecond(library: str, col_name: str, expected: list[int]) -> None:
    df = temporal_dataframe_1(library)
    col = df.__dataframe_namespace__().col
    result = df.select(col(col_name).dt.microsecond())
    result_list = result.collect().get_column_by_name(col_name)
    assert result_list.get_value(0) == expected[0]
    assert result_list.get_value(1) == expected[1]
    assert result_list.get_value(2) == expected[2]


@pytest.mark.parametrize(
    ("col_name", "expected"),
    [
        ("a", [123000, 321000, 987000]),
        ("c", [123543, 321654, 987321]),
        ("e", [123543, 321654, 987321]),
    ],
)
def test_col_microsecond(library: str, col_name: str, expected: list[int]) -> None:
    df = temporal_dataframe_1(library).collect()
    result = df.get_column_by_name(col_name).dt.microsecond()
    assert result.get_value(0) == expected[0]
    assert result.get_value(1) == expected[1]
    assert result.get_value(2) == expected[2]
