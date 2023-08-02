from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import integer_series_6
from tests.utils import interchange_to_pandas


@pytest.mark.parametrize(
    ("ascending", "expected_data"),
    [
        (True, [0, 2, 1]),
        (False, [1, 2, 0]),
    ],
)
def test_column_sorted_indices(
    library: str,
    ascending: bool,
    expected_data: list[int],
    request: pytest.FixtureRequest,
) -> None:
    ser = integer_series_6(library, request)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict(
        {"result": (ser.sorted_indices(ascending=ascending)).rename("result")}
    )
    result_pd = interchange_to_pandas(result, library)["result"]
    # TODO standardise return type?
    result_pd = result_pd.astype("int64")
    expected = pd.Series(expected_data, name="result")
    pd.testing.assert_series_equal(result_pd, expected)
