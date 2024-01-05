from tests.utils import compare_column_with_reference
from tests.utils import integer_dataframe_1


def test_cast_integers(library: str) -> None:
    df = integer_dataframe_1(library)
    pdx = df.__dataframe_namespace__()
    result = df.assign(df.col("a").cast(pdx.Int32()))
    compare_column_with_reference(result.col("a"), [1, 2, 3], dtype=pdx.Int32)
    compare_column_with_reference(result.col("b"), [4, 5, 6], dtype=pdx.Int64)
