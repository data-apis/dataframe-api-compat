from __future__ import annotations

import pandas as pd

from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_column_filter(library: str) -> None:
    df = integer_dataframe_1(library).collect()
    ser = df.get_column_by_name("a")
    mask = ser > 1
    ser = ser.filter(mask)
    result = df.filter(mask)
    result = result.assign(ser.rename("result"))
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series([2, 3], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_expression_filter(library: str) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    ser = namespace.col("a")
    mask = ser > 1
    ser = ser.filter(mask)
    result = df.filter(mask)
    result = result.assign(ser.rename("result"))
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series([2, 3], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_get_rows_by_mask_noop(library: str) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    ser = namespace.col("a")
    mask = ser > 0
    ser = ser.filter(mask)
    result = df.assign(ser.rename("result"))
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series([1, 2, 3], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
