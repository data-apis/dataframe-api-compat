from __future__ import annotations

import pandas as pd

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_get_rows(library: str) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    indices = namespace.column_from_sequence(
        [0, 2, 1], dtype=namespace.Int64(), name="result"
    )
    result = df.get_rows(indices)
    result_pd = interchange_to_pandas(result, library)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [1, 3, 2], "b": [4, 6, 5]})
    pd.testing.assert_frame_equal(result_pd, expected)
