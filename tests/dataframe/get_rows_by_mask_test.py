import pandas as pd
from tests.utils import integer_dataframe_1, convert_dataframe_to_pandas_numpy


def test_get_rows_by_mask(library: str) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    mask = namespace.column_from_sequence([True, False, True], dtype=namespace.Bool())
    result = df.get_rows_by_mask(mask)
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)
    result_pd = convert_dataframe_to_pandas_numpy(result_pd)
    expected = pd.DataFrame({"a": [1, 3], "b": [4, 6]})
    pd.testing.assert_frame_equal(result_pd, expected)
