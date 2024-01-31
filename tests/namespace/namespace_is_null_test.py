from __future__ import annotations

from tests.utils import integer_dataframe_1
from tests.utils import integer_dataframe_2


def test_is_null(library: str) -> None:
    integer_dataframe_1(library)
    other = integer_dataframe_2(library)
    pdx = other.__dataframe_namespace__()
    null = pdx.null
    assert pdx.is_null(null)
    assert not pdx.is_null(float("nan"))
    assert not pdx.is_null(0)
