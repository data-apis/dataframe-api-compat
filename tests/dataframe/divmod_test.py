from __future__ import annotations

from tests.utils import BaseHandler
from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_1


def test_divmod_with_scalar(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    other = 2
    result_quotient, result_remainder = df.__divmod__(other)
    expected_quotient = {"a": [0, 1, 1], "b": [2, 2, 3]}
    expected_remainder = {"a": [1, 0, 1], "b": [0, 1, 0]}
    compare_dataframe_with_reference(result_quotient, expected_quotient, dtype=ns.Int64)
    compare_dataframe_with_reference(result_remainder, expected_remainder, dtype=ns.Int64)
