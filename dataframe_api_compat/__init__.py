from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

__all__ = ["pandas_standard", "polars_standard", "modin_standard"]


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        return importlib.import_module("." + name, __name__)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__version__ = "0.2.7"
