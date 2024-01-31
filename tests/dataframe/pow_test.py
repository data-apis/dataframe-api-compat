from __future__ import annotations

from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_1


def test_float_scalar_powers(library: str) -> None:
    df = integer_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    other = 1.0
    result = df.__pow__(other)
    result = result.cast({"a": pdx.Int64(), "b": pdx.Int64()})
    expected = {"a": [1, 2, 3], "b": [4, 5, 6]}
    compare_dataframe_with_reference(result, expected, dtype=pdx.Int64)
