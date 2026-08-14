"""
Jit-able frozen-dataclass pytrees for JAX: declare a class with jaxtyping
field annotations and @strux.struct derives an immutable, registered,
schema-checked pytree with batch-aware shapes, indexing, pretty printing,
annotations, and serialisation.
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
    astype,
    mapped,
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
    tree_dims,
    tree_getitem,
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
    save,
    to_dict,
)
from strux.inspector import describe

__all__ = [
    "struct",
    "Struct",
    "astype",
    "mapped",
    "schema",
    "Schema",
    "SchemaError",
    "ValidationError",
    "tree_shape",
    "tree_dims",
    "tree_getitem",
    "tree_size",
    "to_str",
    "tree_format",
    "to_dict",
    "from_dict",
    "save",
    "load",
    "describe",
]
