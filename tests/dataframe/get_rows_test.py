from __future__ import annotations

import pytest

from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_1


@pytest.mark.xfail(strict=False)
def test_take(library: str) -> None:
    df = integer_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    df = df.assign((pdx.col("a") - 1).sort(ascending=False).rename("result"))
    pdx = df.__dataframe_namespace__()
    result = df.take(df.col("result"))
    expected = {"a": [3, 2, 1], "b": [6, 5, 4], "result": [0, 1, 2]}
    compare_dataframe_with_reference(result, expected, dtype=pdx.Int64)
