import pandas as pd

from tests.utils import convert_series_to_pandas_numpy, nan_series_1


def test_column_fill_nan(library: str) -> None:
    # todo: test with nullable pandas
    ser = nan_series_1(library)
    namespace = ser.__column_namespace__()
    result = namespace.dataframe_from_dict({"result": ser.fill_nan(-1.0)})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([0.0, 1.0, -1.0], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
