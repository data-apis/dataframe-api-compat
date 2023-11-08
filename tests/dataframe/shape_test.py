from __future__ import annotations

import pytest

from tests.utils import integer_dataframe_1


def test_shape(library: str) -> None:
    df = integer_dataframe_1(library).persist()
    assert df.shape() == (3, 2)

    df = integer_dataframe_1(library)
    with pytest.raises(ValueError):
        df.shape()
