from __future__ import annotations

import pytest

from tests.utils import integer_dataframe_1


def test_shape(library: str, request: pytest.FixtureRequest) -> None:
    if library == "polars-lazy":
        request.node.add_marker(pytest.mark.xfail())
    df = integer_dataframe_1(library)
    assert df.shape() == (3, 2)
