from __future__ import annotations

import pandas as pd

from tests.utils import bool_dataframe_1
from tests.utils import convert_series_to_pandas_numpy
from tests.utils import interchange_to_pandas


def test_column_invert(library: str) -> None:
    df = bool_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    ser = namespace.col("a")
    result = df.insert_column((~ser).rename("result"))
    result_pd = interchange_to_pandas(result, library)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([False, False, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
