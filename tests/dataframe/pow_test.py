from __future__ import annotations

import pandas as pd

from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_float_scalar_powers(library: str) -> None:
    df = integer_dataframe_1(library)
    other = 1.0
    result = df.__pow__(other)
    result_pd = interchange_to_pandas(result).astype(
        {"a": "int64", "b": "int64"},
    )
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pd.testing.assert_frame_equal(result_pd, expected)
