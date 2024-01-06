from __future__ import annotations

from tests.utils import compare_column_with_reference
from tests.utils import nan_dataframe_1


def test_column_fill_nan(library: str) -> None:
    # TODO: test with nullable pandas, check null isn't filled
    df = nan_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    ser = df.col("a")
    result = df.assign(ser.fill_nan(-1.0).rename("result"))
    compare_column_with_reference(result.col("result"), [1.0, 2.0, -1.0], pdx.Float64)


def test_column_fill_nan_with_null(library: str) -> None:
    # TODO: test with nullable pandas, check null isn't filled
    df = nan_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    ser = df.col("a")
    result = df.assign(ser.fill_nan(pdx.null).is_null().rename("result"))
    compare_column_with_reference(result.col("result"), [False, False, True], pdx.Bool)
