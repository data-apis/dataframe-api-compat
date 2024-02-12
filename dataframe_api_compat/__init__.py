from __future__ import annotations

import contextlib

with contextlib.suppress(ModuleNotFoundError):
    from dataframe_api_compat import pandas_standard

with contextlib.suppress(ModuleNotFoundError):
    from dataframe_api_compat import polars_standard

with contextlib.suppress(ModuleNotFoundError):
    from dataframe_api_compat import modin_standard

__all__ = ["pandas_standard", "polars_standard", "modin_standard"]

__version__ = "0.2.6"
