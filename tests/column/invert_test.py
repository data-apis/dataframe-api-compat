from __future__ import annotations

import pandas as pd

from tests.utils import bool_dataframe_1
from tests.utils import interchange_to_pandas


def test_column_invert(library: str) -> None:
    df = bool_dataframe_1(library)
    ser = df.get_column_by_name("a")
    result = df.insert(0, "result", ~ser)
    result_pd = interchange_to_pandas(result, library)["result"]
    expected = pd.Series([False, False, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
