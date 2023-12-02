from __future__ import annotations

from datetime import date
from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from tests.utils import PANDAS_VERSION
from tests.utils import POLARS_VERSION
from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


@pytest.mark.parametrize(
    "pandas_dtype",
    [
        "float64",
        "float32",
        "int64",
        "int32",
        "int16",
        "int8",
        "uint64",
        "uint32",
        "uint16",
        "uint8",
    ],
)
def test_column_from_1d_array(
    library: str,
    pandas_dtype: str,
) -> None:
    ser = integer_dataframe_1(library).col("a").persist()
    namespace = ser.__column_namespace__()
    arr = np.array([1, 2, 3], dtype=pandas_dtype)
    result = namespace.dataframe_from_columns(
        namespace.column_from_1d_array(  # type: ignore[call-arg]
            arr,
            name="result",
        ),
    )
    result_pd = interchange_to_pandas(result)["result"]
    expected = pd.Series([1, 2, 3], name="result", dtype=pandas_dtype)
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_from_1d_array_string(
    library: str,
) -> None:
    ser = integer_dataframe_1(library).persist().col("a")
    namespace = ser.__column_namespace__()
    arr = np.array(["a", "b", "c"])
    result = namespace.dataframe_from_columns(
        namespace.column_from_1d_array(  # type: ignore[call-arg]
            arr,
            name="result",
        ),
    )
    result_pd = interchange_to_pandas(result)["result"]
    expected = pd.Series(["a", "b", "c"], name="result", dtype="object")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_from_1d_array_bool(
    library: str,
) -> None:
    ser = integer_dataframe_1(library).persist().col("a")
    namespace = ser.__column_namespace__()
    arr = np.array([True, False, True])
    result = namespace.dataframe_from_columns(
        namespace.column_from_1d_array(  # type: ignore[call-arg]
            arr,
            name="result",
        ),
    )
    result_pd = interchange_to_pandas(result)["result"]
    expected = pd.Series([True, False, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_datetime_from_1d_array(library: str) -> None:
    ser = integer_dataframe_1(library).persist().col("a")
    namespace = ser.__column_namespace__()
    arr = np.array([date(2020, 1, 1), date(2020, 1, 2)], dtype="datetime64[ms]")
    result = namespace.dataframe_from_columns(
        namespace.column_from_1d_array(  # type: ignore[call-arg]
            arr,
            name="result",
        ),
    )
    result_pd = interchange_to_pandas(result)["result"]
    expected = pd.Series(
        [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        name="result",
        dtype="datetime64[ms]",
    )
    pd.testing.assert_series_equal(result_pd, expected)


@pytest.mark.skipif(
    POLARS_VERSION < (0, 19, 9),
    reason="upstream bug",
)
@pytest.mark.skipif(
    PANDAS_VERSION < (2, 0, 0),
    reason="pandas before non-nano",
)
def test_duration_from_1d_array(library: str) -> None:
    ser = integer_dataframe_1(library).persist().col("a")
    namespace = ser.__column_namespace__()
    arr = np.array([timedelta(1), timedelta(2)], dtype="timedelta64[ms]")
    result = namespace.dataframe_from_columns(
        namespace.column_from_1d_array(  # type: ignore[call-arg]
            arr,
            name="result",
        ),
    )
    if library == "polars-lazy":
        # https://github.com/data-apis/dataframe-api/issues/329
        result_pd = (
            result.dataframe.collect().to_pandas()["result"].astype("timedelta64[ms]")  # type: ignore[attr-defined]
        )
    else:
        result_pd = result.dataframe["result"]  # type: ignore[index]
    expected = pd.Series(
        [timedelta(1), timedelta(2)],
        name="result",
        dtype="timedelta64[ms]",
    )
    pd.testing.assert_series_equal(result_pd, expected)
