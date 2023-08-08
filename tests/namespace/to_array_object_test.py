from __future__ import annotations

import numpy as np
import pytest

from tests.utils import integer_dataframe_1


def test_to_array_object(library: str) -> None:
    df = integer_dataframe_1(library)
    result = np.asarray(df.to_array_object(dtype="int64"))
    expected = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.int64)
    np.testing.assert_array_equal(result, expected)


def test_column_to_array_object(library: str) -> None:
    col = integer_dataframe_1(library).get_column_by_name("a")
    if library == "polars-lazy":
        with pytest.raises(NotImplementedError):
            col.to_array_object(dtype="int64")
        return
    result = np.asarray(col.to_array_object(dtype="int64"))
    result = np.asarray(col.to_array_object(dtype="int64"))
    expected = np.array([1, 2, 3], dtype=np.int64)
    np.testing.assert_array_equal(result, expected)
