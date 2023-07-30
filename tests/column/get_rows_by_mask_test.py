from tests.utils import integer_series_1
from tests.utils import convert_series_to_pandas_numpy
import pandas as pd


def test_column_get_rows_by_mask(library: str) -> None:
    ser = integer_series_1(library)
    namespace = ser.__column_namespace__()
    mask = namespace.column_from_sequence([True, False, True], dtype=namespace.Bool())
    result = ser.get_rows_by_mask(mask)
    result_pd = pd.api.interchange.from_dataframe(
        namespace.dataframe_from_dict({"result": result}).dataframe
    )["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([1, 3], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
