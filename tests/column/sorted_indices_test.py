from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import integer_series_6


@pytest.mark.parametrize(
    ("ascending", "expected_data"),
    [
        (True, [0, 2, 1]),
        (False, [1, 2, 0]),
    ],
)
def test_column_sorted_indices(
    library: str, ascending: bool, expected_data: list[int]
) -> None:
    ser = integer_series_6(library)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict(
        {"result": (ser.sorted_indices(ascending=ascending)).rename("result")}
    )
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    # TODO standardise return type?
    result_pd = result_pd.astype("int64")
    expected = pd.Series(expected_data, name="result")
    pd.testing.assert_series_equal(result_pd, expected)
