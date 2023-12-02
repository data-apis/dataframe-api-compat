import numpy as np
import pandas as pd
import pytest

from tests.utils import integer_dataframe_1
from tests.utils import integer_dataframe_2
from tests.utils import interchange_to_pandas


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
    assert getattr(scalar, attr)(other) == getattr(float_scalar, attr)(other)


def test_float_binary_invalid(library: str) -> None:
    lhs = integer_dataframe_2(library).col("a").mean()
    rhs = integer_dataframe_1(library).col("b").mean()
    with pytest.raises(ValueError):
        _ = lhs > rhs


def test_float_binary_lazy_valid(library: str) -> None:
    df = integer_dataframe_2(library).persist()
    lhs = df.col("a").mean()
    rhs = df.col("b").mean()
    result = lhs > rhs
    assert not bool(result)


@pytest.mark.parametrize(
    "attr",
    [
        "__abs__",
        "__neg__",
    ],
)
def test_float_unary(library: str, attr: str) -> None:
    df = integer_dataframe_2(library).persist()
    with pytest.warns(UserWarning):
        scalar = df.col("a").persist().mean()
    float_scalar = float(scalar)  # type: ignore[arg-type]
    assert getattr(scalar, attr)() == getattr(float_scalar, attr)()


@pytest.mark.parametrize(
    "attr",
    [
        "__int__",
        "__float__",
        "__bool__",
    ],
)
def test_float_unary_invalid(library: str, attr: str) -> None:
    df = integer_dataframe_2(library)
    scalar = df.col("a").mean()
    float_scalar = float(scalar.persist())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        assert getattr(scalar, attr)() == getattr(float_scalar, attr)()


def test_free_standing(library: str) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    ser = namespace.column_from_1d_array(  # type: ignore[call-arg]
        np.array([1, 2, 3]),
        name="a",
    )
    result = float(ser.mean() + 1)  # type: ignore[arg-type]
    assert result == 3.0


def test_right_comparand(library: str) -> None:
    df = integer_dataframe_1(library)
    col = df.col("a")  # [1, 2, 3]
    scalar = df.col("b").get_value(0)  # 4
    result = df.assign((scalar - col).rename("c"))
    result_pd = interchange_to_pandas(result)
    expected = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [3, 2, 1],
        },
    )
    pd.testing.assert_frame_equal(result_pd, expected)
