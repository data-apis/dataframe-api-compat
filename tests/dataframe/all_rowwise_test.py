from __future__ import annotations

from tests.utils import bool_dataframe_1
from tests.utils import compare_dataframe_with_reference


def test_all_horizontal(library: str) -> None:
    df = bool_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    mask = pdx.all_horizontal(*[pdx.col(col_name) for col_name in df.column_names])
    result = df.filter(mask)
    expected = {"a": [True, True], "b": [True, True]}
    compare_dataframe_with_reference(result, expected, dtype=pdx.Bool)
