from __future__ import annotations

import pytest

from tests.utils import integer_dataframe_1


def test_groupby_invalid(library: str) -> None:
    df = integer_dataframe_1(library).select(["a"])
    with pytest.raises((KeyError, TypeError)):
        df.groupby(0)
    with pytest.raises((KeyError, TypeError)):
        df.groupby("0")
    with pytest.raises((KeyError, TypeError)):
        df.groupby(["b"])
