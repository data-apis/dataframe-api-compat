from __future__ import annotations

import pandas as pd

from tests.utils import bool_series_1
from tests.utils import convert_series_to_pandas_numpy
from tests.utils import interchange_to_pandas


def test_column_invert(library: str, request) -> None:
    ser = bool_series_1(library, request)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": (~ser).rename("result")})
    result_pd = interchange_to_pandas(result, library)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([False, True, False], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
