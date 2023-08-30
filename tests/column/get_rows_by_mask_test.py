from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tests.utils import convert_series_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas

if TYPE_CHECKING:
    import pytest


def test_column_get_rows_by_mask(library: str, request: pytest.FixtureRequest) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    ser = namespace.col("a")
    mask = ser > 1
    ser = ser.get_rows_by_mask(mask)
    result = df.get_rows_by_mask(mask)
    result = result.insert_column(ser.rename("result"))
    result_pd = interchange_to_pandas(result, library)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([2, 3], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_get_rows_by_mask_noop(library: str) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    ser = namespace.col("a")
    mask = ser > 0
    ser = ser.get_rows_by_mask(mask)
    result = df.insert_column(ser.rename("result"))
    result_pd = interchange_to_pandas(result, library)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([1, 2, 3], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
