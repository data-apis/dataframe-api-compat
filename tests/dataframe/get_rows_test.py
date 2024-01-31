from __future__ import annotations

import pytest

from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_1


# need to think of a way to test this...
@pytest.mark.xfail()
def test_take(library: str) -> None:
    df = integer_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    result = df.take([2, 1, 0])
    expected = {"a": [3, 2, 1], "b": [6, 5, 4], "result": [0, 1, 2]}
    compare_dataframe_with_reference(result, expected, dtype=pdx.Int64)
