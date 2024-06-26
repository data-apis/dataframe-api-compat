from __future__ import annotations

import pytest
from packaging.version import Version

from tests.utils import BaseHandler
from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_1
from tests.utils import integer_dataframe_2
from tests.utils import pandas_version


def test_join_left(library: BaseHandler) -> None:
    left = integer_dataframe_1(library)
    right = integer_dataframe_2(library).rename({"b": "c"})
    result = left.join(right, left_on="a", right_on="a", how="left")
    ns = result.__dataframe_namespace__()
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [4.0, 2.0, float("nan")]}
    expected_dtype = {
        "a": ns.Int64,
        "b": ns.Int64,
        "c": ns.Int64
        if library.name in ["pandas-nullable", "polars-lazy"]
        else ns.Float64,
    }
    compare_dataframe_with_reference(result, expected, dtype=expected_dtype)  # type: ignore[arg-type]


def test_join_overlapping_names(library: BaseHandler) -> None:
    left = integer_dataframe_1(library)
    right = integer_dataframe_2(library)
    with pytest.raises(ValueError):
        _ = left.join(right, left_on="a", right_on="a", how="left")


def test_join_inner(library: BaseHandler) -> None:
    left = integer_dataframe_1(library)
    right = integer_dataframe_2(library).rename({"b": "c"})
    result = left.join(right, left_on="a", right_on="a", how="inner")
    ns = result.__dataframe_namespace__()
    expected = {"a": [1, 2], "b": [4, 5], "c": [4, 2]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)


@pytest.mark.skip(reason="outer join has changed in Polars recently, need to fixup")
def test_join_outer(library: BaseHandler) -> None:  # pragma: no cover
    left = integer_dataframe_1(library)
    right = integer_dataframe_2(library).rename({"b": "c"})
    result = left.join(right, left_on="a", right_on="a", how="outer").sort("a")
    ns = result.__dataframe_namespace__()
    if (
        library.name == "pandas-nullable" and Version("2.0.0") > pandas_version()
    ):  # pragma: no cover
        # upstream bug
        result = result.cast({"a": ns.Int64()})
    expected = {
        "a": [1, 2, 3, 4],
        "b": [4, 5, 6, float("nan")],
        "c": [4.0, 2.0, float("nan"), 6.0],
    }
    expected_dtype = {
        "a": ns.Int64,
        "b": ns.Int64
        if library.name in ["pandas-nullable", "polars-lazy"]
        else ns.Float64,
        "c": ns.Int64
        if library.name in ["pandas-nullable", "polars-lazy"]
        else ns.Float64,
    }
    compare_dataframe_with_reference(result, expected, dtype=expected_dtype)  # type: ignore[arg-type]


def test_join_two_keys(library: BaseHandler) -> None:
    left = integer_dataframe_1(library)
    right = integer_dataframe_2(library).rename({"b": "c"})
    result = left.join(right, left_on=["a", "b"], right_on=["a", "c"], how="left")
    ns = result.__dataframe_namespace__()
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [4.0, float("nan"), float("nan")]}
    expected_dtype = {
        "a": ns.Int64,
        "b": ns.Int64,
        "c": ns.Int64
        if library.name in ["pandas-nullable", "polars-lazy"]
        else ns.Float64,
    }
    compare_dataframe_with_reference(result, expected, dtype=expected_dtype)  # type: ignore[arg-type]


def test_join_invalid(library: BaseHandler) -> None:
    left = integer_dataframe_1(library)
    right = integer_dataframe_2(library).rename({"b": "c"})
    with pytest.raises(ValueError):
        left.join(right, left_on=["a", "b"], right_on=["a", "c"], how="right")  # type: ignore  # noqa: PGH003
