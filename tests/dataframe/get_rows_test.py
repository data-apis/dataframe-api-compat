from __future__ import annotations

import pandas as pd

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_get_rows(library: str) -> None:
    return None  # todo
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    result = df.insert_column(namespace.col("a").get_rows([0, 2, 1]).rename("result"))
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)["result"]
    expected = pd.Series([1, 3, 2], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
