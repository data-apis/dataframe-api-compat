from __future__ import annotations

from tests.utils import compare_column_with_reference
from tests.utils import integer_dataframe_1


def test_column_divmod(library: str) -> None:
    df = integer_dataframe_1(library).persist()
    pdx = df.__dataframe_namespace__()
    ser = df.get_column("a")
    other = df.get_column("b")
    result_quotient, result_remainder = ser.__divmod__(other)
    # quotient
    result = df.assign(result_quotient.rename("result"))
    compare_column_with_reference(result.get_column("result"), [0, 0, 0], dtype=pdx.Int64)
    # remainder
    result = df.assign(result_remainder.rename("result"))
    compare_column_with_reference(result.get_column("result"), [1, 2, 3], dtype=pdx.Int64)


def test_expression_divmod_with_scalar(library: str) -> None:
    df = integer_dataframe_1(library).persist()
    pdx = df.__dataframe_namespace__()
    ser = df.get_column("a")
    result_quotient, result_remainder = ser.__divmod__(2)
    # quotient
    result = df.assign(result_quotient.rename("result"))
    compare_column_with_reference(result.get_column("result"), [0, 1, 1], dtype=pdx.Int64)
    # remainder
    result = df.assign(result_remainder.rename("result"))
    compare_column_with_reference(result.get_column("result"), [1, 0, 1], dtype=pdx.Int64)
