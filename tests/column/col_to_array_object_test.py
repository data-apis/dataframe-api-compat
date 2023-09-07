from __future__ import annotations

import numpy as np
import pytest

from tests.utils import bool_series_1
from tests.utils import integer_dataframe_1
from tests.utils import integer_series_1


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    ],
)
def test_column_to_array_object(
    library: str, dtype: str, request: pytest.FixtureRequest
) -> None:
    df = integer_dataframe_1(library)
    ser = df.get_column_by_name("a")
    if library == "polars-lazy":
        with pytest.raises(NotImplementedError):
            result = np.asarray(ser.to_array_object(dtype=dtype))
        return
    result = np.asarray(ser.to_array_object(dtype=dtype))
    expected = np.array([1, 2, 3], dtype=np.int64)
    np.testing.assert_array_equal(result, expected)


def test_column_to_array_object_bool(library: str) -> None:
    dtype = "bool"
    df = bool_series_1(library)
    result = np.asarray(df.to_array_object(dtype=dtype))
    expected = np.array([True, False, True], dtype="bool")
    np.testing.assert_array_equal(result, expected)


def test_column_to_array_object_invalid(
    library: str, request: pytest.FixtureRequest
) -> None:
    dtype = "object"
    df = integer_dataframe_1(library)
    with pytest.raises(ValueError):
        np.asarray(df.to_array_object(dtype=dtype))
    with pytest.raises((ValueError, NotImplementedError)):
        np.asarray(df.get_column_by_name("a").to_array_object(dtype=dtype))
    with pytest.raises(ValueError):
        np.asarray(integer_series_1(library).to_array_object(dtype=dtype))
