"""
Jit-able frozen-dataclass pytrees for JAX: declare a class with jaxtyping
field annotations and @strux.struct derives an immutable, registered,
schema-checked pytree with batch-aware shapes, indexing, pretty printing,
annotations, and serialisation.
"""

from strux.schema import (
    Schema,
    SchemaError,
    ValidationError,
    schema,
)
from strux.struct import struct
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

__all__ = [
    "struct",
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
]
