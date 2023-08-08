from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_6
from tests.utils import interchange_to_pandas


def test_unique_indices_column(library: str, request: pytest.FixtureRequest) -> None:
    if library == "polars-lazy":
        request.node.add_marker(pytest.mark.xfail())
    df = integer_dataframe_6(library)
    df = df.get_rows(df.get_column_by_name("b").unique_indices())
    result = df.get_rows(df.sorted_indices())
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [1, 1, 2, 2], "b": [3, 4, 1, 2]})
    pd.testing.assert_frame_equal(result_pd, expected)
