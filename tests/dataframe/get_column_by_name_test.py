from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_get_column_by_name(library: str, request: pytest.FixtureRequest) -> None:
    df = integer_dataframe_1(library)
    result = df.get_column_by_name("a")
    result = df.drop_column("a").insert(0, "a", result)
    df.__dataframe_namespace__()
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_get_column_by_name_invalid(library: str, request: pytest.FixtureRequest) -> None:
    if library == "polars-lazy":
        request.node.add_marker(pytest.mark.xfail())
    df = integer_dataframe_1(library)
    with pytest.raises(ValueError):
        df.get_column_by_name([True, False])
