from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dataframe_api import Column
    from dataframe_api import DataFrame
    from dataframe_api import GroupBy
else:
    Column = object
    DataFrame = object
    GroupBy = object

_ARRAY_API_DTYPES = frozenset(
    (
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    ),
)


LATEST_API_VERSION = "2023.09-beta"
SUPPORTED_VERSIONS = frozenset((LATEST_API_VERSION, "2023.08-beta"))
