from __future__ import annotations

from datetime import date
from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


@pytest.mark.parametrize(
    ("namespace_dtype", "pandas_dtype"),
    [
        ("Float64", "float64"),
        ("Float32", "float32"),
        ("Int64", "int64"),
        ("Int32", "int32"),
        ("Int16", "int16"),
        ("Int8", "int8"),
        ("UInt64", "uint64"),
        ("UInt32", "uint32"),
        ("UInt16", "uint16"),
        ("UInt8", "uint8"),
    ],
)
def test_column_from_1d_array(
    library: str,
    namespace_dtype: str,
    pandas_dtype: str,
) -> None:
    ser = integer_dataframe_1(library).persist().col("a")
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
    [
        ("String", "object"),
    ],
)
def test_column_from_1d_array_string(
    library: str,
    namespace_dtype: str,
    pandas_dtype: str,
) -> None:
    ser = integer_dataframe_1(library).persist().col("a")
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
    [
        ("Bool", "bool"),
    ],
)
def test_column_from_1d_array_bool(
    library: str,
    namespace_dtype: str,
    pandas_dtype: str,
) -> None:
    ser = integer_dataframe_1(library).persist().col("a")
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


def test_datetime_from_1d_array(library: str) -> None:
    ser = integer_dataframe_1(library).persist().col("a")
    namespace = ser.__column_namespace__()
    arr = np.array([date(2020, 1, 1), date(2020, 1, 2)], dtype="datetime64[ms]")
    result = namespace.dataframe_from_columns(
        namespace.column_from_1d_array(
            arr,
            name="result",
            dtype=namespace.Datetime("ms"),
        ),
    )
    result_pd = interchange_to_pandas(result)["result"]
    expected = pd.Series(
        [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        name="result",
        dtype="datetime64[ms]",
    )
    pd.testing.assert_series_equal(result_pd, expected)


def test_duration_from_1d_array(library: str) -> None:
    if library == "polars-lazy":
        # needs fixing upstream
        return
    ser = integer_dataframe_1(library).persist().col("a")
    namespace = ser.__column_namespace__()
    arr = np.array([timedelta(1), timedelta(2)], dtype="timedelta64[ms]")
    result = namespace.dataframe_from_columns(
        namespace.column_from_1d_array(
            arr,
            name="result",
            dtype=namespace.Duration("ms"),
        ),
    )
    result_pd = interchange_to_pandas(result)["result"]
    expected = pd.Series(
        [timedelta(1), timedelta(2)],
        name="result",
        dtype="timedelta64[ms]",
    )
    pd.testing.assert_series_equal(result_pd, expected)
