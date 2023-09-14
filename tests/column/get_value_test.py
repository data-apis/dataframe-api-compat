from __future__ import annotations

import pandas as pd

from tests.utils import integer_dataframe_1
from tests.utils import integer_series_1
from tests.utils import interchange_to_pandas


def test_get_value(library: str) -> None:
    result = integer_series_1(library).get_value(0)
    assert result == 1


def test_get_value_expr(library: str) -> None:
    df = integer_dataframe_1(library)
    col = df.__dataframe_namespace__().col
    result = df.select(col("a").get_value(0))
    result_pd = interchange_to_pandas(result, library)
    expected = pd.DataFrame({"a": [1]})
    pd.testing.assert_frame_equal(result_pd, expected)
