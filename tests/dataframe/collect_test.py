from __future__ import annotations

import pytest

from tests.utils import integer_dataframe_1


def test_collect(library: str) -> None:
    df = integer_dataframe_1(library)
    df = df.persist()
    with pytest.raises(ValueError):
        df.persist()
