import pandas as pd
from tests.utils import bool_series_1
from tests.utils import convert_series_to_pandas_numpy


def test_column_invert(library: str) -> None:
    ser = bool_series_1(library)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": ~ser})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([False, True, False], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
