from __future__ import annotations

from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_1


def test_float_powers_column(library: str) -> None:
    df = integer_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    pdx = df.__dataframe_namespace__()
    ser = pdx.col("a")
    other = pdx.col("b") * 1.0
    result = df.assign(ser.__pow__(other).rename("result"))
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "result": [1.0, 32.0, 729.0]}
    expected_dtype = {"a": pdx.Int64, "b": pdx.Int64, "result": pdx.Float64}
    compare_dataframe_with_reference(result, expected, expected_dtype)  # type: ignore[arg-type]


def test_float_powers_scalar_column(library: str) -> None:
    df = integer_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    pdx = df.__dataframe_namespace__()
    ser = pdx.col("a")
    other = 1.0
    result = df.assign(ser.__pow__(other).rename("result"))
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "result": [1.0, 2.0, 3.0]}
    expected_dtype = {"a": pdx.Int64, "b": pdx.Int64, "result": pdx.Float64}
    compare_dataframe_with_reference(result, expected, expected_dtype)  # type: ignore[arg-type]


def test_int_powers_column(library: str) -> None:
    df = integer_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    pdx = df.__dataframe_namespace__()
    ser = pdx.col("a")
    other = pdx.col("b") * 1
    result = df.assign(ser.__pow__(other).rename("result"))
    if library in ("polars", "polars-lazy"):
        result = result.cast({name: pdx.Int64() for name in ("a", "b", "result")})
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "result": [1, 32, 729]}
    expected_dtype = {name: pdx.Int64 for name in ("a", "b", "result")}
    compare_dataframe_with_reference(result, expected, expected_dtype)


def test_int_powers_scalar_column(library: str) -> None:
    df = integer_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    pdx = df.__dataframe_namespace__()
    ser = pdx.col("a")
    other = 1
    result = df.assign(ser.__pow__(other).rename("result"))
    if library in ("polars", "polars-lazy"):
        result = result.cast({name: pdx.Int64() for name in ("a", "b", "result")})
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "result": [1, 2, 3]}
    expected_dtype = {name: pdx.Int64 for name in ("a", "b", "result")}
    compare_dataframe_with_reference(result, expected, expected_dtype)
