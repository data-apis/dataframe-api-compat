from tests.utils import integer_dataframe_1


def test_parent_dataframe(library: str) -> None:
    df = integer_dataframe_1(library)
    assert pdx.col("a").parent_dataframe is df
