from __future__ import annotations

import pandas as pd

from tests.utils import convert_series_to_pandas_numpy
from tests.utils import integer_series_5


def test_unique_indices_column(library: str) -> None:
    ser = integer_series_5(library)
    namespace = ser.__column_namespace__()
    ser = ser.get_rows(ser.unique_indices())
    ser = ser.get_rows(ser.sorted_indices())
    result = namespace.dataframe_from_dict({"result": ser})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([1, 4], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
