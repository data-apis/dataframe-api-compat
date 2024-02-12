from tests.utils import BaseHandler
from tests.utils import integer_dataframe_1


def test_parent_dataframe(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    assert df.col("a").parent_dataframe is df
