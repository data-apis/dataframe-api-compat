from __future__ import annotations

from typing import Any
from typing import Callable

import pytest

from tests.utils import integer_dataframe_1


@pytest.mark.parametrize("maybe_collect", [lambda x: x, lambda x: x.persist()])
def test_groupby_invalid(library: str, maybe_collect: Callable[[Any], Any]) -> None:
    df = maybe_collect(integer_dataframe_1(library)).select("a")
    with pytest.raises((KeyError, TypeError)):
        df.group_by(0)
    with pytest.raises((KeyError, TypeError)):
        df.group_by("0")
    with pytest.raises((KeyError, TypeError)):
        df.group_by("b")
