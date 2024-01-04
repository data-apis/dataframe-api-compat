from __future__ import annotations

import pandas as pd
import pytest
from packaging.version import parse
from polars.exceptions import SchemaError

from tests.utils import PANDAS_VERSION
from tests.utils import bool_dataframe_2
from tests.utils import integer_dataframe_4
from tests.utils import interchange_to_pandas


@pytest.mark.parametrize(
    ("aggregation", "expected_b", "expected_c"),
    [
        ("any", [True, True], [True, False]),
        ("all", [False, True], [False, False]),
    ],
)
def test_groupby_boolean(
    library: str,
    aggregation: str,
    expected_b: list[bool],
    expected_c: list[bool],
) -> None:
    df = bool_dataframe_2(library)
    df.__dataframe_namespace__()
    result = getattr(df.group_by("key"), aggregation)()
    # need to sort
    result = result.sort("key")
    result_pd = interchange_to_pandas(result)
    if (
        library == "pandas-nullable" and parse("2.0.0") > PANDAS_VERSION
    ):  # pragma: no cover
        # upstream bug
        result_pd = result_pd.astype({"key": "int64"})
    else:
        pass
    expected = pd.DataFrame({"key": [1, 2], "b": expected_b, "c": expected_c})
    pd.testing.assert_frame_equal(result_pd, expected)


def test_group_by_invalid_any_all(library: str) -> None:
    df = integer_dataframe_4(library).persist()
    with pytest.raises((TypeError, SchemaError)):
        df.group_by("key").any()
    with pytest.raises((TypeError, SchemaError)):
        df.group_by("key").all()
