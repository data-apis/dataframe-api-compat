from __future__ import annotations

import pytest

from tests.utils import integer_dataframe_1
from tests.utils import integer_dataframe_2


def test_invalid_comparisons(library: str) -> None:
    with pytest.raises(ValueError):
        _ = integer_dataframe_1(library).col("a") > integer_dataframe_2(library).col("a")


def test_invalid_comparisons_scalar(library: str) -> None:
    with pytest.raises(ValueError):
        _ = (
            integer_dataframe_1(library).col("a")
            > integer_dataframe_2(library).col("a").mean()
        )
