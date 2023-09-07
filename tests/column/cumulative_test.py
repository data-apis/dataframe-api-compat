from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import integer_dataframe_1
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
    df = integer_dataframe_1(library)
    ser = df.get_column_by_name("a")
    expected = pd.Series(expected_data, name="result")
    result = df.insert(0, "result", getattr(ser, func)())
    result_pd = interchange_to_pandas(result, library)["result"]
    pd.testing.assert_series_equal(result_pd, expected)
