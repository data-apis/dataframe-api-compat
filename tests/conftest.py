from typing import Any


def pytest_generate_tests(metafunc: Any) -> None:
    if "library" in metafunc.fixturenames:
        metafunc.parametrize("library", ["pandas-numpy", "pandas-nullable", "polars"])
