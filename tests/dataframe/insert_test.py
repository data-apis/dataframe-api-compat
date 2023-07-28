import pandas as pd
from tests.utils import convert_dataframe_to_pandas_numpy, integer_dataframe_1


def test_insert(library: str) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    new_col = namespace.column_from_sequence([7, 8, 9], dtype=namespace.Int64())
    result = df.insert(1, "c", new_col)
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [1, 2, 3], "c": [7, 8, 9], "b": [4, 5, 6]})
    pd.testing.assert_frame_equal(result_pd, expected)
    # check original df didn't change
    df_pd = pd.api.interchange.from_dataframe(df.dataframe)
    df_pd = convert_dataframe_to_pandas_numpy(df_pd)
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pd.testing.assert_frame_equal(df_pd, expected)