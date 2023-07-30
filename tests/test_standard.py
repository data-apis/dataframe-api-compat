# todo: test that errors are appropriately raised when calls violate standard
from __future__ import annotations

from typing import Any, Callable

import pytest
import pandas as pd
import polars as pl

from tests.utils import (
    convert_series_to_pandas_numpy,
    convert_to_standard_compliant_dataframe,
    float_series_1,
    nan_series_1,
    float_series_3,
    float_series_2,
    float_series_4,
    integer_series_1,
    integer_series_3,
)


@pytest.mark.parametrize(
    ("reduction", "expected"),
    [
        ("min", 1),
        ("max", 3),
        ("sum", 6),
        ("prod", 6),
        ("median", 2.0),
        ("mean", 2.0),
        ("std", 1.0),
        ("var", 1.0),
    ],
)
def test_column_reductions(library: str, reduction: str, expected: float) -> None:
    ser = integer_series_1(library)
    result = getattr(ser, reduction)()
    assert result == expected


def test_column_column() -> None:
    result = (
        convert_to_standard_compliant_dataframe(pl.DataFrame({"a": [1, 2, 3]}))
        .get_column_by_name("a")
        .column
    )
    pd.testing.assert_series_equal(result.to_pandas(), pd.Series([1, 2, 3], name="a"))
    result = (
        convert_to_standard_compliant_dataframe(pd.DataFrame({"a": [1, 2, 3]}))
        .get_column_by_name("a")
        .column
    )
    pd.testing.assert_series_equal(result, pd.Series([1, 2, 3], name="a"))


@pytest.mark.parametrize(
    ("values", "dtype", "expected"),
    [
        ([1, 2, 3], "Int64", pd.Series([1, 2, 3], dtype="int64", name="result")),
        ([1, 2, 3], "Int32", pd.Series([1, 2, 3], dtype="int32", name="result")),
        (
            [1.0, 2.0, 3.0],
            "Float64",
            pd.Series([1, 2, 3], dtype="float64", name="result"),
        ),
        (
            [1.0, 2.0, 3.0],
            "Float32",
            pd.Series([1, 2, 3], dtype="float32", name="result"),
        ),
        (
            [True, False, True],
            "Bool",
            pd.Series([True, False, True], dtype=bool, name="result"),
        ),
    ],
)
def test_column_from_sequence(
    library: str, values: list[Any], dtype: str, expected: pd.Series[Any]
) -> None:
    ser = integer_series_1(library)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict(
        {
            "result": namespace.column_from_sequence(
                values, dtype=getattr(namespace, dtype)()
            )
        }
    )
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_names(library: str) -> None:
    # nameless column
    ser = integer_series_1(library)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": ser})
    assert result.get_column_names() == ["result"]

    # named column
    ser = namespace.column_from_sequence(
        [1, 2, 3], dtype=namespace.Float64(), name="result"
    )
    result = namespace.dataframe_from_dict({"result": ser})
    assert result.get_column_names() == ["result"]

    # named column (different name)
    ser = namespace.column_from_sequence(
        [1, 2, 3], dtype=namespace.Float64(), name="result2"
    )
    with pytest.raises(ValueError):
        namespace.dataframe_from_dict({"result": ser})


def test_fill_null_noop_column(library: str) -> None:
    ser = nan_series_1(library)
    result = ser.fill_null(0)
    # nan should not have changed!
    assert result.column[2] != result.column[2]


@pytest.mark.parametrize(
    ("func", "expected_data"),
    [
        ("cumulative_sum", [1, 3, 6]),
        ("cumulative_prod", [1, 2, 6]),
        ("cumulative_max", [1, 2, 3]),
        ("cumulative_min", [1, 1, 1]),
    ],
)
def test_cumulative_functions_column(
    library: str, func: str, expected_data: list[float]
) -> None:
    ser = integer_series_1(library)
    namespace = ser.__column_namespace__()
    expected = pd.Series(expected_data, name="result")
    result = namespace.dataframe_from_dict({"result": getattr(ser, func)()})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_max(library: str) -> None:
    result = integer_series_1(library).max()
    assert result == 3
