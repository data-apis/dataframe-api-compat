import pandas as pd
import pytest
from dataframe_api_compat.pandas_standard import convert_to_standard_compliant_dataframe


def test_repeated_columns() -> None:
    df = pd.DataFrame({"a": [1, 2]}, index=["b", "b"]).T
    with pytest.raises(
        ValueError, match=r"Expected unique column names, got b 2 time\(s\)"
    ):
        convert_to_standard_compliant_dataframe(df)


def test_non_str_columns() -> None:
    df = pd.DataFrame({0: [1, 2]})
    with pytest.raises(
        TypeError,
        match=r"Expected column names to be of type str, got 0 of type <class 'int'>",
    ):
        convert_to_standard_compliant_dataframe(df)