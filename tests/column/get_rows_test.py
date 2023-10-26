from __future__ import annotations

import pandas as pd

from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_expression_get_rows(library: str) -> None:
    df = integer_dataframe_1(library)
    ser = df.col("a")
    indices = df.col("a") - 1
    result = df.assign(ser.get_rows(indices).rename("result")).select("result")
    result_pd = interchange_to_pandas(result)["result"]
    expected = pd.Series([1, 2, 3], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
