from __future__ import annotations

import sys
from typing import Any

import pytest

from tests.utils import ModinHandler
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
    "modin": ModinHandler("modin"),
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
    if library is None:  # pragma: no cover
        # `LIBRARIES` is already initialized
        return
    else:
        assert library in ("pandas-numpy", "pandas-nullable", "polars-lazy", "modin")
        global LIBRARIES  # noqa: PLW0603
        LIBRARIES = {
            (3, 8): [library],
            (3, 9): [library],
            (3, 10): [library],
            (3, 11): [library],
            (3, 12): [library],
        }


def pytest_generate_tests(metafunc: Any) -> None:
    if "library" in metafunc.fixturenames:
        libraries = LIBRARIES[sys.version_info[:2]]
        lib_handlers = [LIBRARIES_HANDLERS[lib] for lib in libraries]

        metafunc.parametrize("library", lib_handlers, ids=libraries)


ci_skip_ids = [
    # polars does not allow to create a dataframe with non-unique columns
    "non_unique_column_names_test.py::test_repeated_columns[polars-lazy]",
    # it is impossible to create a series with a name different from the string type
    "name_test.py::test_invalid_column_name[polars-lazy]",
]


ci_xfail_ids = [
    # https://github.com/modin-project/modin/issues/7212
    "join_test.py::test_join_left[modin]",
    "join_test.py::test_join_two_keys[modin]",
    "persistedness_test.py::test_cross_df_propagation[modin]",
    # https://github.com/modin-project/modin/issues/3602
    "aggregate_test.py::test_aggregate[modin]",
    "aggregate_test.py::test_aggregate_only_size[modin]",
]


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:  # pragma: no cover
    for item in items:
        if any(id_ in item.nodeid for id_ in ci_xfail_ids):
            item.add_marker(pytest.mark.xfail(strict=True))
        elif any(id_ in item.nodeid for id_ in ci_skip_ids):
            item.add_marker(
                pytest.mark.skip("does not make sense for a specific implementation"),
            )
