from __future__ import annotations

import numpy as np

from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_1


def test_dataframe_from_2d_array(library: str) -> None:
    df = integer_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    pdx = df.__dataframe_namespace__()
    arr = np.array([[1, 4], [2, 5], [3, 6]])
    result = pdx.dataframe_from_2d_array(
        arr,
        names=["a", "b"],
    )
    # TODO: consistent return type, for windows compat?
    result = result.cast({"a": pdx.Int64(), "b": pdx.Int64()})
    expected = {"a": [1, 2, 3], "b": [4, 5, 6]}
    compare_dataframe_with_reference(result, expected, dtype=pdx.Int64)
