from __future__ import annotations

import sys
from typing import Any

LIBRARIES = {
    (3, 8): ["pandas-numpy", "pandas-nullable", "polars-lazy"],
    (3, 9): ["pandas-numpy", "pandas-nullable", "polars-lazy"],
    (3, 10): ["pandas-numpy", "pandas-nullable", "polars-lazy"],
    (3, 11): ["pandas-numpy", "pandas-nullable", "polars-lazy"],
    (3, 12): ["polars-lazy"],
}


def pytest_generate_tests(metafunc: Any) -> None:
    if "library" in metafunc.fixturenames:
        metafunc.parametrize(
            "library",
            LIBRARIES[sys.version_info[:2]],
        )
