# todo: test that errors are appropriately raised when calls violate standard
from __future__ import annotations

from typing import Any, Callable

import pytest
import pandas as pd
import polars as pl

from tests.utils import (
    convert_series_to_pandas_numpy,
    convert_to_standard_compliant_dataframe,
    float_series_1,
    nan_series_1,
    float_series_3,
    float_series_2,
    float_series_4,
    integer_series_1,
    integer_series_3,
)
