from __future__ import annotations

import pandas as pd

from tests.utils import integer_dataframe_1
from tests.utils import integer_series_1
from tests.utils import interchange_to_pandas


def test_column_len(library: str) -> None:
    result = integer_series_1(library).len()
    assert result == 3


def test_expr_len(library: str) -> None:
    df = integer_dataframe_1(library)
    col = df.__dataframe_namespace__().col
    result = df.select(col("a").len())
    result_pd = interchange_to_pandas(result, library)
    expected = pd.DataFrame({"a": [3]})
    if library == "polars-lazy":
        result_pd["a"] = result_pd["a"].astype("int64")
    pd.testing.assert_frame_equal(result_pd, expected)
