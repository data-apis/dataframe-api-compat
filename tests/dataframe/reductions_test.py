from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


@pytest.mark.parametrize(
    ("reduction", "expected"),
    [
        ("min", pd.DataFrame({"a": [1], "b": [4]})),
        ("max", pd.DataFrame({"a": [3], "b": [6]})),
        ("sum", pd.DataFrame({"a": [6], "b": [15]})),
        ("prod", pd.DataFrame({"a": [6], "b": [120]})),
        ("median", pd.DataFrame({"a": [2.0], "b": [5.0]})),
        ("mean", pd.DataFrame({"a": [2.0], "b": [5.0]})),
        ("std", pd.DataFrame({"a": [1.0], "b": [1.0]})),
        ("var", pd.DataFrame({"a": [1.0], "b": [1.0]})),
    ],
)
def test_dataframe_reductions(
    library: str,
    reduction: str,
    expected: pd.DataFrame,
) -> None:
    df = integer_dataframe_1(library)
    result = getattr(df, reduction)()
    result_pd = interchange_to_pandas(result)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    pd.testing.assert_frame_equal(result_pd, expected)
