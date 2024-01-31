from __future__ import annotations

from tests.utils import compare_dataframe_with_reference
from tests.utils import nan_dataframe_1


def test_fill_nan(library: str) -> None:
    df = nan_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = df.fill_nan(-1)
    result = result.cast({"a": ns.Float64()})
    expected = {"a": [1.0, 2.0, -1.0]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Float64)


def test_fill_nan_with_scalar(library: str) -> None:
    df = nan_dataframe_1(library).persist()
    pdx = df.__dataframe_namespace__()
    result = df.fill_nan(pdx.col("a").get_value(0))
    expected = {"a": [1.0, 2.0, 1.0]}
    compare_dataframe_with_reference(result, expected, dtype=pdx.Float64)


def test_fill_nan_with_null(library: str) -> None:
    df = nan_dataframe_1(library).persist()
    pdx = df.__dataframe_namespace__()
    result = df.fill_nan(pdx.null)
    result = result.is_nan().sum()
    result = result.get_column("a").get_value(0).scalar
    if library == "pandas-numpy":
        # null is nan for pandas-numpy
        assert result == 1
    else:
        assert result == 0
