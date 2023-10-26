from __future__ import annotations

import pandas as pd

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_get_rows(library: str) -> None:
    df = integer_dataframe_1(library)
    df.__dataframe_namespace__()
    df = df.assign((df.col("a") - 1).sort(ascending=False).rename("result"))
    df.__dataframe_namespace__()
    result = df.get_rows(df.col("result"))
    result_pd = interchange_to_pandas(result)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [3, 2, 1], "b": [6, 5, 4], "result": [0, 1, 2]})
    pd.testing.assert_frame_equal(result_pd, expected)
