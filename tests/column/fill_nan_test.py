from __future__ import annotations

import pytest

from tests.utils import compare_column_with_reference
from tests.utils import nan_dataframe_1


@pytest.mark.xfail(strict=False)
def test_column_fill_nan(library: str) -> None:
    # TODO: test with nullable pandas, check null isn't filled
    df = nan_dataframe_1(library).persist()
    pdx = df.__dataframe_namespace__()
    ser = pdx.col("a")
    result = df.assign(ser.fill_nan(-1.0).rename("result"))
    expected = [1.0, 2.0, -1.0]
    compare_column_with_reference(
        result.persist().get_column("result"),
        expected,
        dtype=pdx.Float64,
    )


@pytest.mark.xfail(strict=False)
def test_column_fill_nan_with_null(library: str) -> None:
    # TODO: test with nullable pandas, check null isn't filled
    df = nan_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    ser = pdx.col("a")
    result = df.assign(ser.fill_nan(pdx.null).is_null().rename("result"))
    expected = [False, False, True]
    compare_column_with_reference(
        result.persist().get_column("result"),
        expected,
        dtype=pdx.Bool,
    )
