import pandas as pd

from tests.utils import convert_series_to_pandas_numpy, integer_series_1, integer_series_3


def test_column_divmod(library: str) -> None:
    ser = integer_series_1(library)
    other = integer_series_3(library)
    namespace = ser.__column_namespace__()
    result_quotient, result_remainder = ser.__divmod__(other)
    result_quotient_pd = pd.api.interchange.from_dataframe(
        namespace.dataframe_from_dict({"result": result_quotient}).dataframe
    )["result"]
    result_remainder_pd = pd.api.interchange.from_dataframe(
        namespace.dataframe_from_dict({"result": result_remainder}).dataframe
    )["result"]
    expected_quotient = pd.Series([1, 1, 0], name="result")
    expected_remainder = pd.Series([0, 0, 3], name="result")
    result_quotient_pd = convert_series_to_pandas_numpy(result_quotient_pd)
    result_remainder_pd = convert_series_to_pandas_numpy(result_remainder_pd)
    pd.testing.assert_series_equal(result_quotient_pd, expected_quotient)
    pd.testing.assert_series_equal(result_remainder_pd, expected_remainder)


def test_column_divmod_with_scalar(library: str) -> None:
    ser = integer_series_1(library)
    other = 2
    namespace = ser.__column_namespace__()
    result_quotient, result_remainder = ser.__divmod__(other)
    result_quotient_pd = pd.api.interchange.from_dataframe(
        namespace.dataframe_from_dict({"result": result_quotient}).dataframe
    )["result"]
    result_remainder_pd = pd.api.interchange.from_dataframe(
        namespace.dataframe_from_dict({"result": result_remainder}).dataframe
    )["result"]
    expected_quotient = pd.Series([0, 1, 1], name="result")
    expected_remainder = pd.Series([1, 0, 1], name="result")
    result_quotient_pd = convert_series_to_pandas_numpy(result_quotient_pd)
    result_remainder_pd = convert_series_to_pandas_numpy(result_remainder_pd)
    pd.testing.assert_series_equal(result_quotient_pd, expected_quotient)
    pd.testing.assert_series_equal(result_remainder_pd, expected_remainder)
