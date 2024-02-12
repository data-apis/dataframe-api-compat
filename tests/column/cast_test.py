from tests.utils import BaseHandler
from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_1


def test_cast_integers(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = df.assign(df.col("a").cast(ns.Int32()))
    expected = {"a": [1, 2, 3], "b": [4, 5, 6]}
    expected_dtype = {"a": ns.Int32, "b": ns.Int64}
    compare_dataframe_with_reference(result, expected, expected_dtype)
