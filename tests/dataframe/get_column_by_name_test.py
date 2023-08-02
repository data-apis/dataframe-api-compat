from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import convert_series_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_get_column_by_name(library: str, request) -> None:
    if library == "polars-lazy":
        request.node.add_marker(pytest.mark.xfail())
    df = integer_dataframe_1(library)
    result = df.get_column_by_name("a")
    namespace = df.__dataframe_namespace__()
    result = namespace.dataframe_from_dict({"result": (result).rename("result")})
    result_pd = interchange_to_pandas(result, library)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([1, 2, 3], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_get_column_by_name_invalid(library: str, request) -> None:
    if library == "polars-lazy":
        request.node.add_marker(pytest.mark.xfail())
    df = integer_dataframe_1(library)
    with pytest.raises(ValueError):
        df.get_column_by_name([True, False])
