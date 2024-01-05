from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_1


def test_cast_integers(library: str) -> None:
    df = integer_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    result = df.assign(df.col("a").cast(pdx.Int32()))
    compare_dataframe_with_reference(
        result,
        {"a": [1, 2, 3], "b": [4, 5, 6]},
        dtype={"a": pdx.Int32, "b": pdx.Int64},
    )
