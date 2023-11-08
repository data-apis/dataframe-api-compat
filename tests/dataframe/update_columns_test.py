from __future__ import annotations

import pandas as pd

from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_update_columns(library: str) -> None:
    df = integer_dataframe_1(library)
    df.__dataframe_namespace__()
    col = df.col
    result = df.assign(col("a") + 1)
    result_pd = interchange_to_pandas(result)
    expected = pd.DataFrame({"a": [2, 3, 4], "b": [4, 5, 6]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_update_multiple_columns(library: str) -> None:
    df = integer_dataframe_1(library)
    df.__dataframe_namespace__()
    col = df.col
    result = df.assign(col("a") + 1, col("b") + 2)
    result_pd = interchange_to_pandas(result)
    expected = pd.DataFrame({"a": [2, 3, 4], "b": [6, 7, 8]})
    pd.testing.assert_frame_equal(result_pd, expected)
