from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_column_filter(library: str, request: pytest.FixtureRequest) -> None:
    df = integer_dataframe_1(library)
    ser = df.get_column_by_name("a")
    mask = ser > 1
    ser = ser.filter(mask)
    result = df.filter(mask)
    if library == "polars-lazy":
        # created from a different dataframe
        with pytest.raises(ValueError):
            result = result.insert(0, "result", ser)
        return
    result = result.insert(0, "result", ser)
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series([2, 3], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_get_rows_by_mask_noop(library: str) -> None:
    df = integer_dataframe_1(library)
    ser = df.get_column_by_name("a")
    mask = ser > 0
    ser = ser.filter(mask)
    result = df.insert(0, "result", ser)
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series([1, 2, 3], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
