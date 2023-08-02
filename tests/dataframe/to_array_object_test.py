from __future__ import annotations

import numpy as np
import pytest

from tests.utils import integer_dataframe_1


def test_to_array_object(library: str, request) -> None:
    if library == "polars-lazy":
        request.node.add_marker(pytest.mark.xfail())
    df = integer_dataframe_1(library)
    result = np.asarray(df.to_array_object(dtype="int64"))
    expected = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.int64)
    np.testing.assert_array_equal(result, expected)
