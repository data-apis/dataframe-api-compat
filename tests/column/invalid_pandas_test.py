from __future__ import annotations

import pandas as pd
import pytest

from dataframe_api_compat.pandas_standard import convert_to_standard_compliant_dataframe


def test_repeated_columns() -> None:
    df = pd.DataFrame({"a": [1, 2]}, index=["b", "b"]).T
    with pytest.raises(
        ValueError,
        match=r"Expected unique column names, got b 2 time\(s\)",
    ):
        convert_to_standard_compliant_dataframe(df, "2023.08-beta")
