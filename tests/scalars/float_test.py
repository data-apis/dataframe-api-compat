import numpy as np
import pytest

from tests.utils import integer_dataframe_1
from tests.utils import integer_dataframe_2


@pytest.mark.parametrize(
    "attr",
    [
        "__lt__",
        "__le__",
        "__eq__",
        "__ne__",
        "__gt__",
        "__ge__",
        "__add__",
        "__radd__",
        "__sub__",
        "__rsub__",
        "__mul__",
        "__rmul__",
        "__mod__",
        "__rmod__",
        "__pow__",
        "__rpow__",
        "__floordiv__",
        "__rfloordiv__",
        "__truediv__",
        "__rtruediv__",
    ],
)
def test_float_binary(library: str, attr: str) -> None:
    other = 0.5
    df = integer_dataframe_2(library).persist()
    scalar = df.col("a").mean()
    float_scalar = float(scalar)  # type: ignore[arg-type]
    assert getattr(scalar, attr)(other).materialise() == getattr(
        float_scalar,
        attr,
    )(other)


def test_float_binary_invalid(library: str) -> None:
    lhs = integer_dataframe_2(library).persist().col("a").mean()
    rhs = integer_dataframe_1(library).persist().col("b").mean()
    with pytest.raises(ValueError):
        _ = lhs > rhs  # type: ignore[operator]


def test_float_binary_lazy_valid(library: str) -> None:
    df = integer_dataframe_2(library).persist()
    lhs = df.col("a").mean()
    rhs = df.col("b").mean()
    result = lhs > rhs  # type: ignore[operator]
    assert not bool(result)


@pytest.mark.parametrize(
    "attr",
    [
        "__abs__",
        "__int__",
        "__float__",
        "__bool__",
        "__neg__",
    ],
)
def test_float_unary(library: str, attr: str) -> None:
    df = integer_dataframe_2(library).persist()
    scalar = df.col("a").mean()
    float_scalar = float(scalar)  # type: ignore[arg-type]
    assert getattr(scalar, attr)() == getattr(float_scalar, attr)()


def test_free_standing(library: str) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    ser = namespace.column_from_1d_array(
        np.array([1, 2, 3]),
        dtype=namespace.Int64(),
        name="a",
    )
    result = float(ser.mean() + 1)  # type: ignore[operator]
    assert result == 3.0
