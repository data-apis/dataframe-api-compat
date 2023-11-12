from __future__ import annotations

from typing import Any

import polars as pl

from dataframe_api_compat.polars_standard.column_object import Column
from dataframe_api_compat.polars_standard.dataframe_object import DataFrame


class Scalar:
    def __init__(self, value: Any, api_version: str, df: DataFrame | None) -> None:
        self.value = value
        self._api_version = api_version
        self.df = df

    def _validate_other(self, other: Any) -> Any:
        if isinstance(other, (Column, DataFrame)):
            return NotImplemented
        if isinstance(other, Scalar):
            if id(self.df) != id(other.df):
                msg = "cannot compare columns/scalars from different dataframes"
                raise ValueError(
                    msg,
                )
            return other.value
        return other

    def materialise(self) -> Any:
        if self.df is None:
            return pl.select(self.value).item()
        return self.df.materialise_expression(self.value).item()

    def _from_scalar(self, scalar: Scalar) -> Scalar:
        return Scalar(scalar, df=self.df, api_version=self._api_version)

    def __lt__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__lt__(other))

    def __le__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__le__(other))

    def __eq__(self, other: Any) -> Scalar:  # type: ignore[override]
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__eq__(other))

    def __ne__(self, other: Any) -> Scalar:  # type: ignore[override]
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__ne__(other))

    def __gt__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__gt__(other))

    def __ge__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__ge__(other))

    def __add__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__add__(other))

    def __radd__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__radd__(other))

    def __sub__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__sub__(other))

    def __rsub__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__rsub__(other))

    def __mul__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__mul__(other))

    def __rmul__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__rmul__(other))

    def __mod__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__mod__(other))

    def __rmod__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__rmod__(other))

    def __pow__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__pow__(other))

    def __rpow__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__rpow__(other))

    def __floordiv__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__floordiv__(other))

    def __rfloordiv__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__rfloordiv__(other))

    def __truediv__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__truediv__(other))

    def __rtruediv__(self, other: Any) -> Scalar:
        other = self._validate_other(other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self.value.__rtruediv__(other))

    def __neg__(self) -> Scalar:
        return self._from_scalar(self.value.__neg__())

    def __abs__(self) -> Scalar:
        return self._from_scalar(self.value.__abs__())

    def __bool__(self) -> bool:
        item = self.materialise()
        return item.__bool__()  # type: ignore[no-any-return]

    def __int__(self) -> int:
        item = self.materialise()
        return item.__int__()  # type: ignore[no-any-return]

    def __float__(self) -> float:
        item = self.materialise()
        return item.__float__()  # type: ignore[no-any-return]
