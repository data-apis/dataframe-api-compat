from __future__ import annotations

import pandas as pd

from tests.utils import bool_dataframe_1
from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import interchange_to_pandas


def test_all_rowwise(library: str) -> None:
    df = bool_dataframe_1(library)
    df.__dataframe_namespace__()
    result = df.filter(df.all_rowwise())
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [True, True], "b": [True, True]})
    pd.testing.assert_frame_equal(result_pd, expected)
