import numpy as np

from tests.utils import BaseHandler
from tests.utils import compare_column_with_reference
from tests.utils import integer_dataframe_1


def test_free_vs_w_parent(library: BaseHandler) -> None:
    df1 = integer_dataframe_1(library)
    ns = df1.__dataframe_namespace__()
    free_ser1 = ns.column_from_1d_array(  # type: ignore[call-arg]
        np.array([1, 2, 3], dtype="int64"),
        name="preds",
    )
    free_ser2 = ns.column_from_1d_array(  # type: ignore[call-arg]
        np.array([4, 5, 6], dtype="int64"),
        name="preds",
    )

    result = free_ser1 + free_ser2

    compare_column_with_reference(result, [5, 7, 9], dtype=ns.Int64)
    assert ns.is_dtype(result.dtype, "integral")
