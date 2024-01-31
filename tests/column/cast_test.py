from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_1


def test_cast_integers(library: str) -> None:
    df = integer_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    result = df.assign(pdx.col("a").cast(pdx.Int32()))
    expected = {"a": [1, 2, 3], "b": [4, 5, 6]}
    expected_dtype = {"a": pdx.Int32, "b": pdx.Int64}
    compare_dataframe_with_reference(result, expected, expected_dtype)
