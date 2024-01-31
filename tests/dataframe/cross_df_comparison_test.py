from __future__ import annotations

import pytest

from tests.utils import integer_dataframe_1
from tests.utils import integer_dataframe_2


def test_invalid_comparisons(library: str) -> None:
    df1 = integer_dataframe_1(library).persist()
    df2 = integer_dataframe_2(library).persist()
    mask = df2.get_column("a") > 1
    with pytest.raises(ValueError):
        _ = df1.filter(mask)
