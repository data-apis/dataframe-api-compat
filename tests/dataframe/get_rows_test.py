from __future__ import annotations

import pandas as pd

from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_take(library: str) -> None:
    df = integer_dataframe_1(library)
    df.__dataframe_namespace__()
    df = df.assign((df.col("a") - 1).sort(ascending=False).rename("result"))
    df.__dataframe_namespace__()
    result = df.take(df.col("result"))
    result_pd = interchange_to_pandas(result)
    expected = pd.DataFrame({"a": [3, 2, 1], "b": [6, 5, 4], "result": [0, 1, 2]})
    pd.testing.assert_frame_equal(result_pd, expected)
