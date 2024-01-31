import pytest

from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_1
from tests.utils import integer_dataframe_2


def test_cross_df_propagation(library: str) -> None:
    df1 = integer_dataframe_1(library)
    df2 = integer_dataframe_2(library)
    pdx = df1.__dataframe_namespace__()
    df1 = df1 + 1
    df2 = df2.rename({"b": "c"})
    result = df1.join(df2, how="left", left_on="a", right_on="a")
    pdx = result.__dataframe_namespace__()
    expected = {
        "a": [2, 3, 4],
        "b": [5, 6, 7],
        "c": [2.0, float("nan"), 6.0],
    }
    expected_dtype = {
        "a": pdx.Int64,
        "b": pdx.Int64,
        "c": pdx.Int64 if library in ["pandas-nullable", "polars-lazy"] else pdx.Float64,
    }
    compare_dataframe_with_reference(result, expected, dtype=expected_dtype)  # type: ignore[arg-type]


def test_multiple_propagations(library: str) -> None:
    # This is a bit "ugly", as the user is "required" to call `persist`
    # multiple times to do things optimally
    df = integer_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    df = df.persist()
    with pytest.warns(UserWarning):
        df1 = df.filter(pdx.col("a") > 1).persist()
        df2 = df.filter(pdx.col("a") <= 1).persist()
    assert int(df1.get_column("a").mean()) == 2  # type: ignore[call-overload]
    assert int(df2.get_column("a").mean()) == 1  # type: ignore[call-overload]

    # But what if I want to do this
    df = integer_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    df = df.persist()
    df1 = df.filter(pdx.col("a") > 1)
    df2 = df.filter(pdx.col("a") <= 1)

    df1 = df1 + 1
    int(df1.get_column("a").mean())  # type: ignore[call-overload]
    int(df1.get_column("a").mean())  # type: ignore[call-overload]
