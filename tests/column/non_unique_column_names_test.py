import pytest

from tests.utils import BaseHandler


def test_repeated_columns(library: BaseHandler) -> None:
    if library.name in ("pandas-numpy", "pandas-nullable"):
        import pandas as pd

        from dataframe_api_compat.pandas_standard import (
            convert_to_standard_compliant_dataframe,
        )

        df = pd.DataFrame([[1, 2]], columns=["b", "b"])
    elif library.name == "modin":
        import modin.pandas as pd

        from dataframe_api_compat.modin_standard import (
            convert_to_standard_compliant_dataframe,
        )

        df = pd.DataFrame([[1, 2]], columns=["b", "b"])
    with pytest.raises(
        ValueError,
        match=r"Expected unique column names, got b 2 time\(s\)",
    ):
        convert_to_standard_compliant_dataframe(df, "2023.08-beta")
