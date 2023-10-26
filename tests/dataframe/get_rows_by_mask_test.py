from __future__ import annotations

import pandas as pd

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_filter(library: str) -> None:
    df = integer_dataframe_1(library)
    df.__dataframe_namespace__()
    mask = df.col("a") % 2 == 1
    result = df.filter(mask)
    result_pd = interchange_to_pandas(result)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [1, 3], "b": [4, 6]})
    pd.testing.assert_frame_equal(result_pd, expected)
