from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import integer_dataframe_2
from tests.utils import interchange_to_pandas


def test_filter(library: str) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    mask = namespace.column_from_sequence(
        [True, False, True], dtype=namespace.Bool(), name="result"
    )
    result = df.filter(mask)
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [1, 3], "b": [4, 6]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_get_column_by_name_invalid_lazy() -> None:
    df1 = integer_dataframe_1("polars-lazy")
    df2 = integer_dataframe_2("polars-lazy")
    with pytest.raises(
        ValueError, match="Column was created from a different dataframe!"
    ):
        df1.filter(df2.get_column_by_name("a") > 0)
