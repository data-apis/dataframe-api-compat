from __future__ import annotations

import pytest
from packaging.version import Version

from tests.utils import BaseHandler
from tests.utils import convert_to_standard_compliant_dataframe
from tests.utils import integer_dataframe_1
from tests.utils import pandas_version


def test_name(library: BaseHandler) -> None:
    df = integer_dataframe_1(library).persist()
    name = df.col("a").name
    assert name == "a"


def test_pandas_name_if_0_named_column() -> None:
    import pandas as pd

    df = convert_to_standard_compliant_dataframe(pd.DataFrame({0: [1, 2, 3]}))
    assert df.column_names == [0]  # type: ignore[comparison-overlap]
    assert [col.name for col in df.iter_columns()] == [0]  # type: ignore[comparison-overlap]


def test_invalid_column_name(library: BaseHandler) -> None:
    if library.name in ("pandas-numpy", "pandas-nullable"):
        import pandas as pd

        if pandas_version() < Version("2.1.0"):  # pragma: no cover
            pytest.skip(reason="before consoritum standard")
        with pytest.raises(ValueError):
            pd.Series([1, 2, 3], name=0).__column_consortium_standard__()
    elif library.name == "modin":
        import modin.pandas as pd

        with pytest.raises(ValueError):
            pd.Series([1, 2, 3], name=0).__column_consortium_standard__()
    else:  # pragma: no cover
        msg = f"Not supported library: {library}"
        raise AssertionError(msg)
