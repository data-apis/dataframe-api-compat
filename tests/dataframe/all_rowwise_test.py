import pandas as pd

from tests.utils import bool_dataframe_1


def test_all_rowwise(library: str) -> None:
    df = bool_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    result = namespace.dataframe_from_dict({"result": df.all_rowwise()})
    result_pd = pd.api.interchange.from_dataframe(result.dataframe)["result"]
    expected = pd.Series([True, True, False], name="result")
    pd.testing.assert_series_equal(result_pd, expected)
