from __future__ import annotations

from tests.utils import compare_dataframe_with_reference
from tests.utils import nan_dataframe_1


def test_dataframe_is_nan(library: str) -> None:
    df = nan_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    result = df.is_nan()
    expected = {"a": [False, False, True]}
    compare_dataframe_with_reference(result, expected, dtype=pdx.Bool)
