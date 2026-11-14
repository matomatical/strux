"""
Jit-able frozen-dataclass pytrees for JAX: declare a class with jaxtyping
field annotations and @strux.struct derives an immutable registered pytree
with batch-aware shapes, indexing, pretty printing, composable descriptions,
optional checking, and serialisation.
"""

import typing

from strux.schema import (
    Schema,
    SchemaError,
    ValidationError,
    schema,
)
from strux.struct import struct
from strux.annotate import (
    Map,
    Replace,
    Batch,
    check,
)

if typing.TYPE_CHECKING:
    # the Struct form is typing.Annotated to static checkers (an import-as,
    # since Annotated is a typeform and cannot be assigned): checkers read
    # Struct[Env, "batch"] as Env with inert metadata, while at runtime the
    # same expression builds a synthetic isinstance-checkable class
    from typing import Annotated as Struct
else:
    from strux.annotate import Struct
from strux.shapes import (
    as_batch,
    tree_dims,
    tree_getitem,
    tree_iter,
    tree_len,
    tree_shape,
    tree_size,
)
from strux.pprint import (
    to_str,
    tree_format,
)
from strux.serial import (
    from_dict,
    load,
    metadata,
    save,
    to_dict,
)
from strux.inspector import describe

__all__ = [
    "struct",
    "Map",
    "Replace",
    "Batch",
    "check",
    "as_batch",
    "Struct",
    "schema",
    "Schema",
    "SchemaError",
    "ValidationError",
    "tree_shape",
    "tree_dims",
    "tree_getitem",
    "tree_len",
    "tree_iter",
    "tree_size",
    "to_str",
    "tree_format",
    "to_dict",
    "from_dict",
    "metadata",
    "save",
    "load",
    "describe",
]
