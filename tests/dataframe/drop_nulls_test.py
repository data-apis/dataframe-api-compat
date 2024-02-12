from tests.utils import BaseHandler
from tests.utils import compare_dataframe_with_reference
from tests.utils import null_dataframe_1


def test_drop_nulls(library: BaseHandler) -> None:
    df = null_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = df.drop_nulls()
    expected = {"a": [1.0, 2.0]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Float64)
