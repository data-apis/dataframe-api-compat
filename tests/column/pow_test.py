from __future__ import annotations

from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_1


def test_float_powers_column(library: str) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = df.col("b") * 1.0
    result = df.assign(ser.__pow__(other).rename("result"))
    compare_dataframe_with_reference(
        result,
        {"a": [1, 2, 3], "b": [4, 5, 6], "result": [1.0, 32.0, 729.0]},
        {"a": ns.Int64, "b": ns.Int64, "result": ns.Float64},
    )


def test_float_powers_scalar_column(library: str) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = 1.0
    result = df.assign(ser.__pow__(other).rename("result"))
    compare_dataframe_with_reference(
        result,
        {"a": [1, 2, 3], "b": [4, 5, 6], "result": [1.0, 2.0, 3.0]},
        {"a": ns.Int64, "b": ns.Int64, "result": ns.Float64},
    )


def test_int_powers_column(library: str) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = df.col("b") * 1
    result = df.assign(ser.__pow__(other).rename("result"))
    if library in ("polars", "polars-lazy"):
        result = result.cast({name: ns.Int64() for name in ("a", "b", "result")})
    compare_dataframe_with_reference(
        result,
        {"a": [1, 2, 3], "b": [4, 5, 6], "result": [1, 32, 729]},
        {name: ns.Int64 for name in ("a", "b", "result")},
    )


def test_int_powers_scalar_column(library: str) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = 1
    result = df.assign(ser.__pow__(other).rename("result"))
    if library in ("polars", "polars-lazy"):
        result = result.cast({name: ns.Int64() for name in ("a", "b", "result")})
    compare_dataframe_with_reference(
        result,
        {"a": [1, 2, 3], "b": [4, 5, 6], "result": [1, 2, 3]},
        {name: ns.Int64 for name in ("a", "b", "result")},
    )
