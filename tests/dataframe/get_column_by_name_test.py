from __future__ import annotations

import pandas as pd

from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_get_column(library: str) -> None:
    df = integer_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    result = df.assign(pdx.col("a").rename("_tmp")).drop("a").rename({"_tmp": "a"})
    result_pd = interchange_to_pandas(result)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})[["b", "a"]]
    pd.testing.assert_frame_equal(result_pd, expected)
