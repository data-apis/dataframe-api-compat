from __future__ import annotations

import pandas as pd

from tests.utils import interchange_to_pandas
from tests.utils import nan_dataframe_1


def test_column_is_nan(library: str) -> None:
    df = nan_dataframe_1(library).persist()
    ser = df.col("a")
    result = df.assign(ser.is_nan().rename("result"))
    result_pd = interchange_to_pandas(result)["result"]
    expected = pd.Series([False, False, True], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
