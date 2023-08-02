from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tests.utils import convert_series_to_pandas_numpy
from tests.utils import integer_series_5
from tests.utils import interchange_to_pandas

if TYPE_CHECKING:
    import pytest


def test_unique_indices_column(library: str, request: pytest.FixtureRequest) -> None:
    ser = integer_series_5(library, request)
    namespace = ser.__column_namespace__()
    ser = ser.get_rows(ser.unique_indices())
    ser = ser.get_rows(ser.sorted_indices())
    result = namespace.dataframe_from_dict({"result": (ser).rename("result")})
    result_pd = interchange_to_pandas(result, library)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([1, 4], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
