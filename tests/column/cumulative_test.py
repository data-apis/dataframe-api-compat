from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import convert_series_to_pandas_numpy
from tests.utils import integer_series_1
from tests.utils import interchange_to_pandas


@pytest.mark.parametrize(
    ("func", "expected_data"),
    [
        ("cumulative_sum", [1, 3, 6]),
        ("cumulative_prod", [1, 2, 6]),
        ("cumulative_max", [1, 2, 3]),
        ("cumulative_min", [1, 1, 1]),
    ],
)
def test_cumulative_functions_column(
    library: str, func: str, expected_data: list[float], request: pytest.FixtureRequest
) -> None:
    ser = integer_series_1(library, request)
    namespace = ser.__column_namespace__()
    expected = pd.Series(expected_data, name="result")
    result = namespace.dataframe_from_dict(
        {"result": (getattr(ser, func)()).rename("result")}
    )
    result_pd = interchange_to_pandas(result, library)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    pd.testing.assert_series_equal(result_pd, expected)
