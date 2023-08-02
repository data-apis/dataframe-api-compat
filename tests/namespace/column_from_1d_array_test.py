from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.utils import convert_series_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_column_from_1d_array(library: str, request: pytest.FixtureRequest) -> None:
    if library == "polars-lazy":
        request.node.add_marker(pytest.mark.xfail())
    ser = integer_dataframe_1(library).get_column_by_name("a")
    namespace = ser.__column_namespace__()
    arr = np.array([1, 2, 3])
    result = namespace.dataframe_from_dict(
        {
            "result": namespace.column_from_1d_array(
                arr, name="result", dtype=namespace.Int64()
            )
        }
    )
    result_pd = interchange_to_pandas(result, library)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([1, 2, 3], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
