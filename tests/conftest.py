from __future__ import annotations

import sys
from typing import Any

from tests.utils import PandasHandler
from tests.utils import PolarsHandler

LIBRARIES = {
    (3, 8): ["pandas-numpy", "pandas-nullable", "polars-lazy"],
    (3, 9): ["pandas-numpy", "pandas-nullable", "polars-lazy"],
    (3, 10): ["pandas-numpy", "pandas-nullable", "polars-lazy"],
    (3, 11): ["pandas-numpy", "pandas-nullable", "polars-lazy"],
    (3, 12): ["polars-lazy"],
}

LIBRARIES_HANDLERS = {
    "pandas-numpy": PandasHandler("pandas-numpy"),
    "pandas-nullable": PandasHandler("pandas-nullable"),
    "polars-lazy": PolarsHandler("polars-lazy"),
}


def pytest_addoption(parser: Any) -> None:
    parser.addoption(
        "--library",
        action="store",
        default=None,
        type=str,
        help="library to test",
    )


def pytest_configure(config: Any) -> None:
    library = config.option.library
    if library is None:
        # `LIBRARIES` is already initialized
        return
    else:
        assert library in ("pandas-numpy", "pandas-nullable", "polars-lazy")
        global LIBRARIES  # noqa: PLW0603
        LIBRARIES = {
            (3, 8): [library],
            (3, 9): [library],
            (3, 10): [library],
            (3, 11): [library],
            (3, 12): [library],
        }

    # TODO: potential way to add filterwarnings for different librariess: config.addinivalue_line
    # for "filterwarnings", "ignore:Ray execution environment not yet initialized:UserWarning"


def pytest_generate_tests(metafunc: Any) -> None:
    if "library" in metafunc.fixturenames:
        libraries = LIBRARIES[sys.version_info[:2]]
        lib_handlers = [LIBRARIES_HANDLERS[lib] for lib in libraries]

        metafunc.parametrize("library", lib_handlers, ids=libraries)
