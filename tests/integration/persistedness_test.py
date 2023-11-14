import pandas as pd
import pytest

from tests.utils import convert_dataframe_to_pandas_numpy
from tests.utils import integer_dataframe_1
from tests.utils import integer_dataframe_2
from tests.utils import interchange_to_pandas


def test_within_df_propagation(library: str) -> None:
    df1 = integer_dataframe_1(library)
    df1 = df1
    df1 = df1 + 1
    with pytest.raises(ValueError):
        _ = int(df1.col("a").mean())

    df1 = integer_dataframe_1(library)
    df1 = df1.persist()
    df1 = df1 + 1
    # the call below would recompute `df1 + 1` multiple times
    with pytest.raises(ValueError):
        _ = int(df1.col("a").mean())

    # this is the correct way
    df1 = integer_dataframe_1(library)
    df1 = df1 + 1
    df1 = df1.persist()
    assert int(df1.col("a").mean()) == 3


def test_within_df_within_col_propagation(library: str) -> None:
    df1 = integer_dataframe_1(library)
    df1 = df1 + 1
    df1 = df1.persist()
    assert int((df1.col("a") + 1).mean()) == 4


def test_cross_df_propagation(library: str) -> None:
    df1 = integer_dataframe_1(library)
    df2 = integer_dataframe_2(library)
    df1 = df1 + 1
    df2 = df2.rename_columns({"b": "c"}).persist()
    result = df1.join(df2, how="left", left_on="a", right_on="a")
    result_pd = convert_dataframe_to_pandas_numpy(interchange_to_pandas(result))
    expected = pd.DataFrame(
        {
            "a": [2, 3, 4],
            "b": [5, 6, 7],
            "c": [2.0, float("nan"), 6.0],
        },
    )
    pd.testing.assert_frame_equal(result_pd, expected)


def test_multiple_propagations(library: str) -> None:
    # This is a bit "ugly", as the user is "required" to call `persist`
    # multiple times to do things optimally
    df = integer_dataframe_1(library)
    df = df.persist()
    df1 = df.filter(df.col("a") > 1).persist()
    df2 = df.filter(df.col("a") <= 1).persist()
    assert int(df1.col("a").mean()) == 2
    assert int(df2.col("a").mean()) == 1

    # But what if I want to do this
    df = integer_dataframe_1(library)
    df = df.persist()
    df1 = df.filter(df.col("a") > 1)
    df2 = df.filter(df.col("a") <= 1)

    df1 = df1 + 1
    # without this persist, `df1 + 1` will be computed twice
    df1 = df1.persist()
    int(df1.col("a").mean())
    int(df1.col("b").mean())


def test_parent_propagations(library: str) -> None:
    # Set up something like this:
    #
    #         df
    #     df1    df2
    #
    # If I persist df1, then that triggers df.
    # If I then want call some scalar on df2, that will have to trigger
    # df again. If df2 wasn't persisted, then df would be recomputed.
    # So, we need to persist df2 as well.
    df = integer_dataframe_1(library)
    df1 = df.filter(df.col("a") > 1)
    df2 = df.filter(df.col("a") <= 1)

    df1 = df1.persist()
    with pytest.raises(ValueError):
        int(df2.col("a").mean())