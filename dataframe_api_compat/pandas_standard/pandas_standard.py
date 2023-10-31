from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dataframe_api import Column
    from dataframe_api import DataFrame
else:
    Column = object
    DataFrame = object
    Namespace = object
    Aggregation = object


class Null:
    ...


null = Null()

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

if TYPE_CHECKING:
    from dataframe_api import Aggregation
    from dataframe_api import Column
    from dataframe_api import DataFrame
    from dataframe_api import GroupBy
else:
    Column = object
    DataFrame = object
    GroupBy = object
    Namespace = object
    Aggregation = object
