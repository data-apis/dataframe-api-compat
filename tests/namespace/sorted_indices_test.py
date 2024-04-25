from __future__ import annotations

from tests.utils import BaseHandler
from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_6


def test_column_sorted_indices_ascending(library: BaseHandler) -> None:
    df = integer_dataframe_6(library)
    ns = df.__dataframe_namespace__()
    sorted_indices = df.col("b").sorted_indices()
    result = df.assign(sorted_indices.rename("result"))
    expected_1 = {
        "a": [1, 1, 1, 2, 2],
        "b": [4, 4, 3, 1, 2],
        "result": [3, 4, 2, 0, 1],
    }
    expected_2 = {
        "a": [1, 1, 1, 2, 2],
        "b": [4, 4, 3, 1, 2],
        "result": [3, 4, 2, 1, 0],
    }
    if library.name == "polars-lazy":
        result = result.cast({"result": ns.Int64()})
    try:
        compare_dataframe_with_reference(result, expected_1, dtype=ns.Int64)
    except AssertionError:  # pragma: no cover
        # order isn't determinist, so try both
        compare_dataframe_with_reference(result, expected_2, dtype=ns.Int64)


def test_column_sorted_indices_descending(library: BaseHandler) -> None:
    df = integer_dataframe_6(library)
    ns = df.__dataframe_namespace__()
    sorted_indices = df.col("b").sorted_indices(ascending=False)
    result = df.assign(sorted_indices.rename("result"))
    expected_1 = {
        "a": [1, 1, 1, 2, 2],
        "b": [4, 4, 3, 1, 2],
        "result": [1, 0, 2, 4, 3],
    }
    expected_2 = {
        "a": [1, 1, 1, 2, 2],
        "b": [4, 4, 3, 1, 2],
        "result": [0, 1, 2, 4, 3],
    }
    if library.name == "polars-lazy":
        result = result.cast({"result": ns.Int64()})
    try:
        compare_dataframe_with_reference(result, expected_1, dtype=ns.Int64)
    except AssertionError:
        # order isn't determinist, so try both
        compare_dataframe_with_reference(result, expected_2, dtype=ns.Int64)
