from __future__ import annotations

import contextlib
import math
from abc import abstractmethod
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Mapping

from packaging.version import Version
from packaging.version import parse

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from dataframe_api import Column
    from dataframe_api import DataFrame
    from dataframe_api.typing import DType


def pandas_version() -> Version:
    import pandas as pd

    return parse(pd.__version__)


def polars_version() -> Version:
    import polars as pl

    return parse(pl.__version__)


class BaseHandler:
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def create_dataframe(
        self,
        data: Any,
        api_version: str | None = None,
    ) -> DataFrame:
        ...


class PandasHandler(BaseHandler):
    # for `pandas-nullable` case
    # https://pandas.pydata.org/docs/user_guide/basics.html#dtypes
    mapping: ClassVar[dict[str, str]] = {
        "bool": "boolean",
        "int64": "Int64",
        "float64": "Float64",
    }

    def __init__(self, name: str) -> None:
        assert name in ("pandas-numpy", "pandas-nullable")
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def create_dataframe(
        self,
        data: Any,
        api_version: str | None = None,
    ) -> DataFrame:
        import pandas as pd

        import dataframe_api_compat.pandas_standard

        df = pd.DataFrame(data)
        if self.name == "pandas-nullable":
            new_dtypes = {
                col_name: self.mapping.get(str(dtype), str(dtype))
                for col_name, dtype in zip(df.columns, df.dtypes)
            }
            df = df.astype(new_dtypes)

        return (
            dataframe_api_compat.pandas_standard.convert_to_standard_compliant_dataframe(
                df,
                api_version=api_version or "2023.11-beta",
            )
        )


class PolarsHandler(BaseHandler):
    def __init__(self, name: str) -> None:
        assert name == "polars-lazy"
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def create_dataframe(
        self,
        data: Any,
        api_version: str | None = None,
    ) -> DataFrame:
        import polars as pl

        import dataframe_api_compat.polars_standard

        df = pl.DataFrame(data)

        return (
            dataframe_api_compat.polars_standard.convert_to_standard_compliant_dataframe(
                df,
                api_version=api_version or "2023.11-beta",
            )
        )


class ModinHandler(BaseHandler):
    def __init__(self, name: str) -> None:
        assert name == "modin"
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def create_dataframe(
        self,
        data: Any,
        api_version: str | None = None,
    ) -> DataFrame:
        import modin.pandas as pd

        import dataframe_api_compat.modin_standard

        df = pd.DataFrame(data)

        return (
            dataframe_api_compat.modin_standard.convert_to_standard_compliant_dataframe(
                df,
                api_version=api_version or "2023.11-beta",
            )
        )


def convert_to_standard_compliant_dataframe(
    df: pd.DataFrame | pl.DataFrame,
    api_version: str | None = None,
) -> DataFrame:
    import pandas as pd

    try:
        polars_installed = True
        import polars as pl
    except ModuleNotFoundError:
        polars_installed = False

    if isinstance(df, pd.DataFrame):
        import dataframe_api_compat.pandas_standard

        return (
            dataframe_api_compat.pandas_standard.convert_to_standard_compliant_dataframe(
                df,
                api_version=api_version,
            )
        )
    elif polars_installed and isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        import dataframe_api_compat.polars_standard

        df_lazy = df.lazy() if isinstance(df, pl.DataFrame) else df
        return (
            dataframe_api_compat.polars_standard.convert_to_standard_compliant_dataframe(
                df_lazy,
                api_version=api_version,
            )
        )
    else:  # pragma: no cover
        msg = f"Got unexpected type: {type(df)}"
        raise AssertionError(msg)


def integer_dataframe_1(
    library: BaseHandler,
    api_version: str | None = None,
) -> DataFrame:
    return library.create_dataframe(
        {"a": [1, 2, 3], "b": [4, 5, 6]},
        api_version=api_version,
    )


def integer_dataframe_2(library: BaseHandler) -> DataFrame:
    return library.create_dataframe(
        {"a": [1, 2, 4], "b": [4, 2, 6]},
    )


def integer_dataframe_3(library: BaseHandler) -> DataFrame:
    return library.create_dataframe(
        {"a": [1, 2, 3, 4, 5, 6, 7], "b": [7, 6, 5, 4, 3, 2, 1]},
    )


def integer_dataframe_4(library: BaseHandler) -> DataFrame:
    return library.create_dataframe(
        {"key": [1, 1, 2, 2], "b": [1, 2, 3, 4], "c": [4, 5, 6, 7]},
    )


def integer_dataframe_5(
    library: BaseHandler,
    api_version: str | None = None,
) -> DataFrame:
    return library.create_dataframe(
        {"a": [1, 1], "b": [4, 3]},
        api_version=api_version,
    )


def integer_dataframe_6(
    library: BaseHandler,
    api_version: str | None = None,
) -> DataFrame:
    return library.create_dataframe(
        {"a": [1, 1, 1, 2, 2], "b": [4, 4, 3, 1, 2]},
        api_version=api_version,
    )


def integer_dataframe_7(library: BaseHandler) -> DataFrame:
    return library.create_dataframe({"a": [1, 2, 3], "b": [1, 2, 4]})


def nan_dataframe_1(library: BaseHandler) -> DataFrame:
    if library.name == "pandas-nullable":
        import pandas as pd

        df = pd.DataFrame({"a": [1.0, 2.0, 0.0]}, dtype="Float64")
        other = pd.DataFrame({"a": [1.0, 1.0, 0.0]}, dtype="Float64")
        return convert_to_standard_compliant_dataframe(df / other)
    return library.create_dataframe({"a": [1.0, 2.0, float("nan")]})


def nan_dataframe_2(library: BaseHandler) -> DataFrame:
    if library.name == "pandas-nullable":
        import pandas as pd

        df = pd.DataFrame({"a": [0.0, 1.0, 0.0]}, dtype="Float64")
        other = pd.DataFrame({"a": [1.0, 1.0, 0.0]}, dtype="Float64")
        return convert_to_standard_compliant_dataframe(df / other)
    return library.create_dataframe({"a": [0.0, 1.0, float("nan")]})


def null_dataframe_1(library: BaseHandler) -> DataFrame:
    if library.name == "pandas-nullable":
        import pandas as pd

        df = pd.DataFrame({"a": [1.0, 2.0, pd.NA]}, dtype="Float64")
        return convert_to_standard_compliant_dataframe(df)
    if library.name == "polars-lazy":
        import polars as pl

        df = pl.DataFrame({"a": [1.0, 2.0, None]})
        return convert_to_standard_compliant_dataframe(df)
    return library.create_dataframe({"a": [1.0, 2.0, float("nan")]})


def null_dataframe_2(library: BaseHandler) -> DataFrame:
    if library.name == "pandas-nullable":
        import pandas as pd

        df = pd.DataFrame(
            {"a": [1.0, 0.0, pd.NA], "b": [1.0, 1.0, pd.NA]},
            dtype="Float64",
        )
        return convert_to_standard_compliant_dataframe(df / df)
    if library.name == "polars-lazy":
        import polars as pl

        df = pl.DataFrame({"a": [1.0, float("nan"), None], "b": [1.0, 1.0, None]})
        return convert_to_standard_compliant_dataframe(df)
    return library.create_dataframe(
        {"a": [1.0, -1.0, float("nan")], "b": [1.0, -1.0, float("nan")]},
    )


def bool_dataframe_1(
    library: BaseHandler,
    api_version: str = "2023.09-beta",
) -> DataFrame:
    return library.create_dataframe(
        {"a": [True, True, False], "b": [True, True, True]},
        api_version=api_version,
    )


def bool_dataframe_2(library: BaseHandler) -> DataFrame:
    return library.create_dataframe(
        {
            "key": [1, 1, 2, 2],
            "b": [False, True, True, True],
            "c": [True, False, False, False],
        },
    )


def bool_dataframe_3(library: BaseHandler) -> DataFrame:
    return library.create_dataframe(
        {"a": [False, False], "b": [False, True], "c": [True, True]},
    )


def float_dataframe_1(library: BaseHandler) -> DataFrame:
    return library.create_dataframe({"a": [2.0, 3.0]})


def float_dataframe_2(library: BaseHandler) -> DataFrame:
    return library.create_dataframe({"a": [2.0, 1.0]})


def float_dataframe_3(library: BaseHandler) -> DataFrame:
    if library.name == "pandas-nullable":
        import pandas as pd

        df = pd.DataFrame({"a": [0.0, 2.0]}, dtype="Float64")
        other = pd.DataFrame({"a": [0.0, 1.0]}, dtype="Float64")
        return convert_to_standard_compliant_dataframe(df / other)
    return library.create_dataframe({"a": [float("nan"), 2.0]})


def temporal_dataframe_1(library: BaseHandler) -> DataFrame:
    if library.name in ["pandas-numpy", "pandas-nullable"]:
        import pandas as pd

        df = pd.DataFrame(
            {
                # the data for column "a" differs from other implementations due to pandas 1.5 compat
                # https://github.com/data-apis/dataframe-api-compat/commit/aeca5cf1a052033b72388e3f87ad8b70d66cedec
                "a": [
                    datetime(2020, 1, 1, 1, 2, 1, 123000),
                    datetime(2020, 1, 2, 3, 1, 2, 321000),
                    datetime(2020, 1, 3, 5, 4, 9, 987000),
                ],
                "b": [
                    timedelta(1, milliseconds=1),
                    timedelta(2, milliseconds=3),
                    timedelta(3, milliseconds=5),
                ],
                "c": [
                    datetime(2020, 1, 1, 1, 2, 1, 123543),
                    datetime(2020, 1, 2, 3, 1, 2, 321654),
                    datetime(2020, 1, 3, 5, 4, 9, 987321),
                ],
                "d": [
                    timedelta(1, milliseconds=1),
                    timedelta(2, milliseconds=3),
                    timedelta(3, milliseconds=5),
                ],
                "e": [
                    datetime(2020, 1, 1, 1, 2, 1, 123543),
                    datetime(2020, 1, 2, 3, 1, 2, 321654),
                    datetime(2020, 1, 3, 5, 4, 9, 987321),
                ],
                "f": [
                    timedelta(1, milliseconds=1),
                    timedelta(2, milliseconds=3),
                    timedelta(3, milliseconds=5),
                ],
                "index": [0, 1, 2],
            },
        ).astype(
            {
                "a": "datetime64[ms]",
                "b": "timedelta64[ms]",
                "c": "datetime64[us]",
                "d": "timedelta64[us]",
                "e": "datetime64[ns]",
                "f": "timedelta64[ns]",
            },
        )
        return convert_to_standard_compliant_dataframe(df)
    if library.name == "polars-lazy":
        import polars as pl

        df = pl.DataFrame(
            {
                "a": [
                    datetime(2020, 1, 1, 1, 2, 1, 123543),
                    datetime(2020, 1, 2, 3, 1, 2, 321654),
                    datetime(2020, 1, 3, 5, 4, 9, 987321),
                ],
                "b": [
                    timedelta(1, milliseconds=1),
                    timedelta(2, milliseconds=3),
                    timedelta(3, milliseconds=5),
                ],
                "c": [
                    datetime(2020, 1, 1, 1, 2, 1, 123543),
                    datetime(2020, 1, 2, 3, 1, 2, 321654),
                    datetime(2020, 1, 3, 5, 4, 9, 987321),
                ],
                "d": [
                    timedelta(1, milliseconds=1),
                    timedelta(2, milliseconds=3),
                    timedelta(3, milliseconds=5),
                ],
                "e": [
                    datetime(2020, 1, 1, 1, 2, 1, 123543),
                    datetime(2020, 1, 2, 3, 1, 2, 321654),
                    datetime(2020, 1, 3, 5, 4, 9, 987321),
                ],
                "f": [
                    timedelta(1, milliseconds=1),
                    timedelta(2, milliseconds=3),
                    timedelta(3, milliseconds=5),
                ],
                "index": [0, 1, 2],
            },
            schema={
                "a": pl.Datetime("ms"),
                "b": pl.Duration("ms"),
                "c": pl.Datetime("us"),
                "d": pl.Duration("us"),
                "e": pl.Datetime("ns"),
                "f": pl.Duration("ns"),
                "index": pl.Int64,
            },
        )
        return convert_to_standard_compliant_dataframe(df)

    return library.create_dataframe(
        {
            "a": [
                datetime(2020, 1, 1, 1, 2, 1, 123000),
                datetime(2020, 1, 2, 3, 1, 2, 321000),
                datetime(2020, 1, 3, 5, 4, 9, 987000),
            ],
            "b": [
                timedelta(1, milliseconds=1),
                timedelta(2, milliseconds=3),
                timedelta(3, milliseconds=5),
            ],
            "c": [
                datetime(2020, 1, 1, 1, 2, 1, 123543),
                datetime(2020, 1, 2, 3, 1, 2, 321654),
                datetime(2020, 1, 3, 5, 4, 9, 987321),
            ],
            "d": [
                timedelta(1, milliseconds=1),
                timedelta(2, milliseconds=3),
                timedelta(3, milliseconds=5),
            ],
            "e": [
                datetime(2020, 1, 1, 1, 2, 1, 123543),
                datetime(2020, 1, 2, 3, 1, 2, 321654),
                datetime(2020, 1, 3, 5, 4, 9, 987321),
            ],
            "f": [
                timedelta(1, milliseconds=1),
                timedelta(2, milliseconds=3),
                timedelta(3, milliseconds=5),
            ],
            "index": [0, 1, 2],
        },
    )


def compare_column_with_reference(
    column: Column,
    reference: list[Any],
    dtype: Any,
) -> None:
    with contextlib.suppress(UserWarning):
        # the comparison should work regardless of whether method `persist` has already been called or not
        column = column.persist()
    col_len = column.len().scalar
    assert col_len == len(reference), f"column length: {col_len} != {len(reference)}"
    assert isinstance(
        column.dtype,
        dtype,
    ), f"column dtype: {column.dtype} isn't a instance of {dtype}"
    for idx in range(col_len):
        a, b = reference[idx], column.get_value(idx).scalar
        if a == b:
            return

        # copied from pandas
        rtol, atol = 1e-5, 1e-8
        assert math.isclose(
            a,
            b,
            rel_tol=rtol,
            abs_tol=atol,
        ), f"expected {a:.5f} but got {b:.5f}, with rtol={rtol}, atol={atol}"


def compare_dataframe_with_reference(
    dataframe: DataFrame,
    reference: Mapping[str, list[Any]],
    dtype: Any | Mapping[str, Any],
) -> None:
    assert dataframe.column_names == list(
        reference.keys(),
    ), f"dataframe column names: '{dataframe.column_names}' != '{list(reference.keys())}'"
    for col_name in dataframe.column_names:
        col_dtype = dtype[col_name] if isinstance(dtype, dict) else dtype
        compare_column_with_reference(
            dataframe.col(col_name),
            reference[col_name],
            dtype=col_dtype,
        )


def mixed_dataframe_1(library: BaseHandler) -> DataFrame:
    df: Any
    data = {
        "a": [1, 2, 3],
        "b": [1, 2, 3],
        "c": [1, 2, 3],
        "d": [1, 2, 3],
        "e": [1, 2, 3],
        "f": [1, 2, 3],
        "g": [1, 2, 3],
        "h": [1, 2, 3],
        "i": [1.0, 2.0, 3.0],
        "j": [1.0, 2.0, 3.0],
        "k": [True, False, True],
        "l": ["a", "b", "c"],
        "m": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
        "n": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
        "o": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
        "p": [timedelta(days=1), timedelta(days=2), timedelta(days=3)],
        "q": [timedelta(days=1), timedelta(days=2), timedelta(days=3)],
    }
    if library.name == "pandas-nullable":
        import pandas as pd

        df = pd.DataFrame(data).astype(
            {
                "a": "Int64",
                "b": "Int32",
                "c": "Int16",
                "d": "Int8",
                "e": "UInt64",
                "f": "UInt32",
                "g": "UInt16",
                "h": "UInt8",
                "i": "Float64",
                "j": "Float32",
                "k": "bool",
                "l": "string[pyarrow]",
                "m": "datetime64[s]",
                "n": "datetime64[ms]",
                "o": "datetime64[us]",
                "p": "timedelta64[ms]",
                "q": "timedelta64[us]",
            },
        )
        return convert_to_standard_compliant_dataframe(df)

    result = library.create_dataframe(data)
    ns = result.__dataframe_namespace__()
    dtypes: Mapping[str, DType] = {
        "a": ns.Int64(),
        "b": ns.Int32(),
        "c": ns.Int16(),
        "d": ns.Int8(),
        "e": ns.UInt64(),
        "f": ns.UInt32(),
        "g": ns.UInt16(),
        "h": ns.UInt8(),
        "i": ns.Float64(),
        "j": ns.Float32(),
        "k": ns.Bool(),
        "l": ns.String(),
        "m": ns.Datetime("ms"),
        "n": ns.Datetime("ms"),
        "o": ns.Datetime("us"),
        "p": ns.Duration("ms"),
        "q": ns.Duration("us"),
    }
    return result.cast(dtypes)
