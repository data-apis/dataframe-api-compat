from __future__ import annotations

from tests.utils import BaseHandler
from tests.utils import integer_dataframe_1


def test_column_len(library: BaseHandler) -> None:
    result = integer_dataframe_1(library).col("a").n_unique().persist().scalar
    assert result == 3
