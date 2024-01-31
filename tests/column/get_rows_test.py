from __future__ import annotations

import pytest

from tests.utils import compare_column_with_reference
from tests.utils import integer_dataframe_1


@pytest.mark.xfail(strict=False)
def test_expression_take(library: str) -> None:
    df = integer_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    ser = pdx.col("a")
    indices = pdx.col("a") - 1
    result = df.assign(ser.take(indices).rename("result")).select("result")
    compare_column_with_reference(result.col("result"), [1, 2, 3], dtype=pdx.Int64)
