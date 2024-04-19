import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal

from tests.utils import BaseHandler
from tests.utils import compare_dataframe_with_reference
from tests.utils import float_dataframe_1
from tests.utils import integer_dataframe_1


def test_shift_with_fill_value(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = df.assign(df.col("a").shift(1).fill_null(999))
    if library.name == "pandas-numpy":
        result = result.cast({name: ns.Int64() for name in ("a", "b")})
    expected = {"a": [999, 1, 2], "b": [4, 5, 6]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)


def test_shift_without_fill_value(library: BaseHandler) -> None:
    df = float_dataframe_1(library)
    result = df.assign(df.col("a").shift(-1))
    if library.name == "pandas-numpy":
        expected = pd.DataFrame({"a": [3.0, float("nan")]})
        pd.testing.assert_frame_equal(result.dataframe, expected)
    elif library.name == "pandas-nullable":
        expected = pd.DataFrame({"a": [3.0, None]}, dtype="Float64")
        pd.testing.assert_frame_equal(result.dataframe, expected)
    elif library.name == "polars-lazy":
        expected = pl.DataFrame({"a": [3.0, None]})
        assert_frame_equal(result.dataframe.collect(), expected)  # type: ignore[attr-defined]
    else:  # pragma: no cover
        msg = "unexpected library"
        raise AssertionError(msg)


def test_shift_with_fill_value_complicated(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = df.assign(df.col("a").shift(1).fill_null(df.col("a").mean()))
    if library.name == "pandas-nullable":
        result = result.cast({"a": ns.Float64()})
    expected = {"a": [2.0, 1, 2], "b": [4, 5, 6]}
    expected_dtype = {"a": ns.Float64, "b": ns.Int64}
    compare_dataframe_with_reference(result, expected, expected_dtype)  # type: ignore[arg-type]
