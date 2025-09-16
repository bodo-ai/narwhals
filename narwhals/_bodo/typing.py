from __future__ import annotations  # pragma: no cover

from typing import (
    TYPE_CHECKING,  # pragma: no cover
    Union,  # pragma: no cover
)

if TYPE_CHECKING:
    import sys
    from typing import Literal, TypeVar

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias

    from narwhals._bodo.dataframe import BodoDataFrame, BodoLazyFrame
    from narwhals._bodo.expr import BodoExpr
    from narwhals._bodo.series import BodoSeries

    IntoBodoExpr: TypeAlias = Union[BodoExpr, BodoSeries]
    FrameT = TypeVar("FrameT", BodoDataFrame, BodoLazyFrame)
    NativeAccessor: TypeAlias = Literal[
        "arr", "cat", "dt", "list", "meta", "name", "str", "bin", "struct"
    ]
