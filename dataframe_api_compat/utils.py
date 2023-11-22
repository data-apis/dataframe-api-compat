from typing import Any


# Technically, it would be possible to correctly type hint this function
# with a tonne of overloads, but for now, it' not worth it, just use Any
def validate_comparand(left: Any, right: Any) -> Any:
    if hasattr(left, "__dataframe_namespace__") and hasattr(
        right,
        "__column_namespace__",
    ):
        if right.parent_dataframe is not None and right.parent_dataframe is not left:
            msg = "Cannot combine columns from different dataframes"
            raise ValueError(msg)
        return right.column
    if hasattr(left, "__column_namespace__") and hasattr(right, "__column_namespace__"):
        if (
            right.parent_dataframe is not None
            and right.parent_dataframe is not left.parent_dataframe
        ):
            msg = "Cannot combine columns from different dataframes"
            raise ValueError(msg)
        return right.column
    if hasattr(left, "__dataframe_namespace__") and hasattr(
        right,
        "__scalar_namespace__",
    ):
        if right.parent_dataframe is not None and right.parent_dataframe is not left:
            msg = "Cannot combine columns from different dataframes"
            raise ValueError(msg)
        return right.scalar
    if hasattr(left, "__column_namespace__") and hasattr(right, "__scalar_namespace__"):
        if (
            right.parent_dataframe is not None
            and right.parent_dataframe is not left.parent_dataframe
        ):
            msg = "Cannot combine columns from different dataframes"
            raise ValueError(msg)
        return right.scalar
    if hasattr(left, "__scalar_namespace__") and hasattr(right, "__scalar_namespace__"):
        if (
            right.parent_dataframe is not None
            and right.parent_dataframe is not left.parent_dataframe
        ):
            msg = "Cannot combine columns from different dataframes"
            raise ValueError(msg)
        return right.scalar

    # Not implemented: return NotImplemeted so that the other object can try to handle it
    if hasattr(left, "__scalar_namespace__") and hasattr(right, "__column_namespace__"):
        return NotImplemented
    if hasattr(left, "__scalar_namespace__") and hasattr(
        right,
        "__dataframe_namespace__",
    ):
        return NotImplemented
    if hasattr(left, "__column_namespace__") and hasattr(
        right,
        "__dataframe_namespace__",
    ):
        return NotImplemented

    return right
