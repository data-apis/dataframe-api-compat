from __future__ import annotations

import pandas as pd

from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_mean(library: str) -> None:
    df = integer_dataframe_1(library)
    df.__dataframe_namespace__()
    result = df.assign((df.col("a") - df.col("a").mean()).rename("result"))
    result_pd = interchange_to_pandas(result)["result"]
    expected = pd.Series([-1, 0, 1.0], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
