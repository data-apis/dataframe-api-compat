from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from packaging.version import Version
from packaging.version import parse

from tests.utils import BaseHandler
from tests.utils import bool_dataframe_2
from tests.utils import compare_dataframe_with_reference
from tests.utils import integer_dataframe_4


@pytest.mark.parametrize(
    ("aggregation", "expected_b", "expected_c"),
    [
        ("any", [True, True], [True, False]),
        ("all", [False, True], [False, False]),
    ],
)
def test_groupby_boolean(
    library: BaseHandler,
    aggregation: str,
    expected_b: list[bool],
    expected_c: list[bool],
) -> None:
    df = bool_dataframe_2(library)
    ns = df.__dataframe_namespace__()
    result = getattr(df.group_by("key"), aggregation)()
    # need to sort
    result = result.sort("key")
    if library.name == "pandas-nullable" and parse(pd.__version__) < Version(
        "2.0.0",
    ):  # pragma: no cover
        # upstream bug
        result = result.cast({"key": ns.Int64()})
    expected = {"key": [1, 2], "b": expected_b, "c": expected_c}
    expected_dtype = {"key": ns.Int64, "b": ns.Bool, "c": ns.Bool}
    compare_dataframe_with_reference(result, expected, dtype=expected_dtype)  # type: ignore[arg-type]


def test_group_by_invalid_any_all(library: BaseHandler) -> None:
    df = integer_dataframe_4(library).persist()

    exceptions: tuple[Any, ...] = (TypeError,)
    if library.name == "polars-lazy":
        from polars.exceptions import SchemaError

        exceptions = (TypeError, SchemaError)
    with pytest.raises(exceptions):
        df.group_by("key").any()
    with pytest.raises(exceptions):
        df.group_by("key").all()
