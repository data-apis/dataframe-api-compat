from __future__ import annotations

import pytest

from tests.utils import BaseHandler
from tests.utils import compare_dataframe_with_reference
from tests.utils import nan_dataframe_1


def test_fill_nan(library: BaseHandler) -> None:
    df = nan_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = df.fill_nan(-1)
    result = result.cast({"a": ns.Float64()})
    expected = {"a": [1.0, 2.0, -1.0]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Float64)


def test_fill_nan_with_scalar(library: BaseHandler) -> None:
    df = nan_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = df.fill_nan(df.col("a").get_value(0))
    result = result.cast({"a": ns.Float64()})
    expected = {"a": [1.0, 2.0, 1.0]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Float64)


def test_fill_nan_with_scalar_invalid(library: BaseHandler) -> None:
    df = nan_dataframe_1(library)
    other = df + 1
    with pytest.raises(ValueError):
        _ = df.fill_nan(other.col("a").get_value(0))


def test_fill_nan_with_null(library: BaseHandler) -> None:
    df = nan_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = df.fill_nan(ns.null)
    n_nans = result.is_nan().sum()
    result = n_nans.col("a").persist().get_value(0).scalar
    if library.name in ("pandas-numpy", "modin"):
        # null is nan for pandas-numpy
        assert result == 1
    else:
        assert result == 0
