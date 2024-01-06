from __future__ import annotations

from tests.utils import bool_dataframe_1
from tests.utils import compare_column_with_reference


def test_expression_invert(library: str) -> None:
    df = bool_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    ser = df.col("a")
    result = df.assign((~ser).rename("result"))
    compare_column_with_reference(result.col("result"), [False, False, True], pdx.Bool)


def test_column_invert(library: str) -> None:
    df = bool_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    ser = df.col("a")
    result = df.assign((~ser).rename("result"))
    compare_column_with_reference(result.col("result"), [False, False, True], pdx.Bool)
