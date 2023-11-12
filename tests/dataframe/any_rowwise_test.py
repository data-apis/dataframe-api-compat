from __future__ import annotations

import pandas as pd
import pytest

from tests.utils import bool_dataframe_1
from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import interchange_to_pandas


def test_any_rowwise(library: str) -> None:
    df = bool_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    mask = namespace.any_rowwise(*[df.col(col_name) for col_name in df.column_names])  # type: ignore[attr-defined]
    result = df.filter(mask)
    result_pd = interchange_to_pandas(result)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [True, True, False], "b": [True, True, True]})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_any_rowwise_invalid(library: str) -> None:
    df = bool_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    with pytest.raises(ValueError):
        _ = namespace.any_rowwise(df.col("a"), (df + 1).col("b"))  # type: ignore[attr-defined]
