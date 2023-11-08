from __future__ import annotations

import pandas as pd

from tests.utils import interchange_to_pandas
from tests.utils import nan_dataframe_1
from tests.utils import null_dataframe_1


def test_column_is_null_1(library: str) -> None:
    df = nan_dataframe_1(library).persist()
    ser = df.col("a")
    result = df.assign(ser.is_null().rename("result"))
    result_pd = interchange_to_pandas(result)["result"]
    if library == "pandas-numpy":
        expected = pd.Series([False, False, True], name="result")
    else:
        expected = pd.Series([False, False, False], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_is_null_2(library: str) -> None:
    df = null_dataframe_1(library).persist()
    ser = df.col("a")
    result = df.assign(ser.is_null().rename("result"))
    result_pd = interchange_to_pandas(result)["result"]
    expected = pd.Series([False, False, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
