import pandas as pd
import pytest

from tests.utils import convert_series_to_pandas_numpy, integer_dataframe_1


def test_get_column_by_name(library: str) -> None:
    df = integer_dataframe_1(library)
    result = df.get_column_by_name("a")
    namespace = df.__dataframe_namespace__()
    result = namespace.dataframe_from_dict({"result": result})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    result_pd = convert_series_to_pandas_numpy(result_pd)
    expected = pd.Series([1, 2, 3], name="result")
    pd.testing.assert_series_equal(result_pd, expected)


def test_get_column_by_name_invalid(library: str) -> None:
    df = integer_dataframe_1(library)
    with pytest.raises(ValueError):
        df.get_column_by_name([True, False])
