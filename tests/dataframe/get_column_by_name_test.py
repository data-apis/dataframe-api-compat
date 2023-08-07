from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_get_column_by_name(library: str) -> None:
    df = integer_dataframe_1(library)
    result = df.get_column_by_name("a").rename("_tmp")
    result = df.insert(0, "_tmp", result).drop_column("a").rename_columns({"_tmp": "a"})
    df.__dataframe_namespace__()
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_get_column_by_name_invalid(library: str) -> None:
    df = integer_dataframe_1(library)
    with pytest.raises((ValueError, TypeError)):
        df.get_column_by_name([True, False])
