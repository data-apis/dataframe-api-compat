from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.utils import integer_dataframe_1, interchange_to_pandas


@pytest.mark.parametrize(
    ("namespace_dtype", "pandas_dtype"),
    [],
)
def test_column_from_1d_array(
    library: str,
    namespace_dtype: str,
    pandas_dtype: str,
) -> None:
    ser = integer_dataframe_1(library).collect().col("a")
    namespace = ser.__column_namespace__()
    arr = np.array([1, 2, 3])
    result = namespace.dataframe_from_columns(
        namespace.column_from_1d_array(
            arr,
            name="result",
            dtype=getattr(namespace, namespace_dtype)(),
        ),
    )
    result_pd = interchange_to_pandas(result)["result"]
    expected = pd.Series([1, 2, 3], name="result", dtype=pandas_dtype)
    pd.testing.assert_series_equal(result_pd, expected)


@pytest.mark.parametrize(
    ("namespace_dtype", "pandas_dtype"),
    [],
)
def test_column_from_1d_array_string(
    library: str,
    namespace_dtype: str,
    pandas_dtype: str,
) -> None:
    ser = integer_dataframe_1(library).collect().col("a")
    namespace = ser.__column_namespace__()
    arr = np.array(["a", "b", "c"])
    result = namespace.dataframe_from_columns(
        namespace.column_from_1d_array(
            arr,
            name="result",
            dtype=getattr(namespace, namespace_dtype)(),
        ),
    )
    result_pd = interchange_to_pandas(result)["result"]
    expected = pd.Series(["a", "b", "c"], name="result", dtype=pandas_dtype)
    pd.testing.assert_series_equal(result_pd, expected)


@pytest.mark.parametrize(
    ("namespace_dtype", "pandas_dtype"),
    [],
)
def test_column_from_1d_array_bool(
    library: str,
    namespace_dtype: str,
    pandas_dtype: str,
) -> None:
    ser = integer_dataframe_1(library).collect().col("a")
    namespace = ser.__column_namespace__()
    arr = np.array([True, False, True])
    result = namespace.dataframe_from_columns(
        namespace.column_from_1d_array(
            arr,
            name="result",
            dtype=getattr(namespace, namespace_dtype)(),
        ),
    )
    result_pd = interchange_to_pandas(result)["result"]
    expected = pd.Series([True, False, True], name="result", dtype=pandas_dtype)
    pd.testing.assert_series_equal(result_pd, expected)


# def test_datetime_from_1d_array(library: str) -> None:
#         namespace.column_from_1d_array(
#             arr,
#         ),


# def test_duration_from_1d_array(library: str) -> None:
#     if library == "polars-lazy":
#         # needs fixing upstream
#         namespace.column_from_1d_array(
#             arr,
#         ),
