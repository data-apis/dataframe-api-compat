from tests.utils import integer_dataframe_1


def test_shape(library: str) -> None:
    df = integer_dataframe_1(library)
    assert df.shape() == (3, 2)
