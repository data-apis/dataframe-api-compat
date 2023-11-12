import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal

from tests.utils import float_dataframe_1
from tests.utils import integer_dataframe_1
from tests.utils import interchange_to_pandas


def test_shift_with_fill_value(library: str) -> None:
    df = integer_dataframe_1(library)
    result = df.assign(df.col("a").shift(1, fill_value=999))  # type: ignore[attr-defined]
    expected = pd.DataFrame(
        {
            "a": [999, 1, 2],
            "b": [4, 5, 6],
        },
    )
    result_pd = interchange_to_pandas(result)
    pd.testing.assert_frame_equal(result_pd, expected)


def test_shift_without_fill_value(library: str) -> None:
    df = float_dataframe_1(library)
    result = df.assign(df.col("a").shift(-1))
    if library == "pandas-numpy":
        expected = pd.DataFrame({"a": [3.0, float("nan")]})
        pd.testing.assert_frame_equal(result.dataframe, expected)
    elif library == "pandas-nullable":
        expected = pd.DataFrame({"a": [3.0, None]}, dtype="Float64")
        pd.testing.assert_frame_equal(result.dataframe, expected)
    elif library == "polars-lazy":
        expected = pl.DataFrame({"a": [3.0, None]})
        assert_frame_equal(result.dataframe.collect(), expected)
    else:  # pragma: no cover
        msg = "unexpected library"
        raise AssertionError(msg)


def test_shift_with_fill_value_complicated(library: str) -> None:
    df = integer_dataframe_1(library)
    result = df.assign(df.col("a").shift(1, fill_value=df.col("a").min()))  # type: ignore[attr-defined]
    expected = pd.DataFrame(
        {
            "a": [1, 1, 2],
            "b": [4, 5, 6],
        },
    )
    result_pd = interchange_to_pandas(result)
    pd.testing.assert_frame_equal(result_pd, expected)