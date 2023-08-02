from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from tests.utils import convert_series_to_pandas_numpy
from tests.utils import integer_series_1
from tests.utils import integer_series_3


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
    request: pytest.FixtureRequest,
) -> None:
    ser: Any
    ser = integer_series_1(library, request)
    other = integer_series_3(library, request)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict(
        {"result": (getattr(ser, comparison)(other)).rename("result")}
    )
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    expected = pd.Series(expected_data, name="result")
    result_pd = convert_series_to_pandas_numpy(result_pd)
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
    request: pytest.FixtureRequest,
) -> None:
    ser: Any
    ser = integer_series_1(library, request)
    other = 3
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict(
        {"result": (getattr(ser, comparison)(other)).rename("result")}
    )
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    expected = pd.Series(expected_data, name="result")
    if library == "polars" and comparison == "__pow__":
        # todo: fix
        result_pd = result_pd.astype("int64")
    result_pd = convert_series_to_pandas_numpy(result_pd)
    pd.testing.assert_series_equal(result_pd, expected)
