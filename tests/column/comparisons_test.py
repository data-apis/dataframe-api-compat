from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from tests.utils import integer_dataframe_1
from tests.utils import integer_dataframe_7
from tests.utils import interchange_to_pandas


@pytest.mark.parametrize(
    ("comparison", "expected_data"),
    [
        ("__eq__", [True, True, False]),
        ("__ne__", [False, False, True]),
        ("__ge__", [True, True, False]),
        ("__gt__", [False, False, False]),
        ("__le__", [True, True, True]),
        ("__lt__", [False, False, True]),
        ("__add__", [2, 4, 7]),
        ("__sub__", [0, 0, -1]),
        ("__mul__", [1, 4, 12]),
        ("__truediv__", [1, 1, 0.75]),
        ("__floordiv__", [1, 1, 0]),
        ("__pow__", [1, 4, 81]),
        ("__mod__", [0, 0, 3]),
    ],
)
def test_column_comparisons(
    library: str,
    comparison: str,
    expected_data: list[object],
) -> None:
    ser: Any
    df = integer_dataframe_7(library)
    ser = df.get_column_by_name("a")
    other = df.get_column_by_name("b")
    result = df.insert(0, "result", (getattr(ser, comparison)(other)))
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series(expected_data, name="result")
    pd.testing.assert_series_equal(result_pd, expected)


@pytest.mark.parametrize(
    ("comparison", "expected_data"),
    [
        ("__eq__", [False, False, True]),
        ("__ne__", [True, True, False]),
        ("__ge__", [False, False, True]),
        ("__gt__", [False, False, False]),
        ("__le__", [True, True, True]),
        ("__lt__", [True, True, False]),
        ("__add__", [4, 5, 6]),
        ("__sub__", [-2, -1, 0]),
        ("__mul__", [3, 6, 9]),
        ("__truediv__", [1 / 3, 2 / 3, 1]),
        ("__floordiv__", [0, 0, 1]),
        ("__pow__", [1, 8, 27]),
        ("__mod__", [1, 2, 0]),
    ],
)
def test_column_comparisons_scalar(
    library: str,
    comparison: str,
    expected_data: list[object],
) -> None:
    ser: Any
    df = integer_dataframe_1(library)
    ser = df.get_column_by_name("a")
    other = 3
    result = df.insert(0, "result", (getattr(ser, comparison)(other)))
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series(expected_data, name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_invalid_comparison() -> None:
    df1 = integer_dataframe_1("polars-lazy")
    df2 = integer_dataframe_1("polars-lazy")
    with pytest.raises(ValueError):
        _ = df1.get_column_by_name("a") > df2.get_column_by_name("a")
