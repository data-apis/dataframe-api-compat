from __future__ import annotations

import pandas as pd

from tests.utils import bool_dataframe_1
from tests.utils import interchange_to_pandas


def test_expression_invert(library: str) -> None:
    df = bool_dataframe_1(library)
    df.__dataframe_namespace__()
    ser = df.col("a")
    result = df.assign((~ser).rename("result"))
    result_pd = interchange_to_pandas(result)["result"]
    expected = pd.Series([False, False, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_column_invert(library: str) -> None:
    df = bool_dataframe_1(library).persist()
    ser = df.col("a")
    result = df.assign((~ser).rename("result"))
    result_pd = interchange_to_pandas(result)["result"]
    expected = pd.Series([False, False, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
