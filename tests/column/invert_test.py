from __future__ import annotations

from tests.utils import bool_dataframe_1
from tests.utils import compare_column_with_reference


def test_expression_invert(library: str) -> None:
    df = bool_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    pdx = df.__dataframe_namespace__()
    ser = pdx.col("a")
    result = df.assign((~ser).rename("result"))
    expected = [False, False, True]
    compare_column_with_reference(
        result.persist().get_column("result"),
        expected,
        dtype=pdx.Bool,
    )


def test_column_invert(library: str) -> None:
    df = bool_dataframe_1(library).persist()
    pdx = df.__dataframe_namespace__()
    ser = pdx.col("a")
    result = df.assign((~ser).rename("result"))
    expected = [False, False, True]
    compare_column_with_reference(result.get_column("result"), expected, dtype=pdx.Bool)
