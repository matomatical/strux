import dataclasses

import jax
import jax.numpy as jnp
import numpy as np


def struct(Class, static_fieldnames=()):
    """
    Transform a class into an immutable dataclass that is also registered as a
    JAX PyTree. Intended to be used as a wrapper, as in:

    ```
    @strux.struct
    class MyDataClass:
        field1: int
        # etc.
    ```
    """
    # wrap class as an immutable Python dataclass
    Dataclass = dataclasses.dataclass(Class, frozen=True)

    # decide which fields are data vs. static
    fields = [field.name for field in dataclasses.fields(Dataclass)]
    data_fields = [name for name in fields if name not in static_fieldnames]
    meta_fields = [name for name in fields if name in static_fieldnames]
    missing_fields = set(static_fieldnames) - set(meta_fields)
    if missing_fields:
        raise ValueError(f"Invalid static_fieldnames {missing_fields}")
    
    # register dataclass as a JAX pytree node
    jax.tree_util.register_dataclass(
        nodetype=Dataclass,
        data_fields=data_fields,
        meta_fields=meta_fields,
    )
    
    # overwrite string render methods to use custom pretty printing.
    Dataclass.__str__ = to_str
    Dataclass.__format__ = format
    
    # add some other convenience methods
    Dataclass.replace = dataclasses.replace
    Dataclass.size = property(lambda self: size(self))

    # allow type indexing
    # TODO: make this work properly with jaxtyping
    Dataclass.__class_getitem__ = classmethod(lambda cls, _: cls)
    return Dataclass


def to_str(
    tree,
    indent: str = "  ",
    max_depth: int | None = None,
) -> str:
    """
    Construct a multi-line string representation of the contents and shape of a
    PyTree.

    Inputs:

    * tree: PyTree.
        The PyTree to render as a string.
    * indent: str (default two spaces).
        String to use for indenting the levels. For example, you could replace
        the default "  " with " " or "    " or "\\t" to your liking.
    * max_depth: optional int.
        Replace contents below this level of nesting as '...'. For example,
        max_depth=1 will replace all children of the root node by '...'.

    Note: Accepts most built-in PyTree components as well as structs, but does
    not work for all PyTrees.
    """
    lines = []
    def _put(s: str, depth: int):
        lines.append(indent * depth + s)
    def _walk(tree, prefix: str, suffix: str, depth: int):
        if dataclasses.is_dataclass(tree):
            if depth == max_depth:
                _put(f"{prefix}{type(tree).__name__}(...){suffix}", depth=depth)
            else:
                _put(f"{prefix}{type(tree).__name__}(", depth=depth)
                state = tree.__getstate__() or {}
                for field, value in state.items():
                    _walk(value, prefix=f"{field}=", suffix=",", depth=depth+1)
                _put(f"){suffix}", depth=depth)
        elif isinstance(tree, tuple):
            if depth == max_depth:
                _put(f"{prefix}(...){suffix}", depth=depth)
            else:
                _put(f"{prefix}(", depth=depth)
                for item in tree:
                    _walk(item, prefix="", suffix=",", depth=depth+1)
                _put(f"){suffix}", depth=depth)
        elif isinstance(tree, list):
            if depth == max_depth:
                _put(f"{prefix}[...]{suffix}", depth=depth)
            else:
                _put(f"{prefix}[", depth=depth)
                for item in tree:
                    _walk(item, prefix="", suffix=",", depth=depth+1)
                _put(f"]{suffix}", depth=depth)
        elif isinstance(tree, dict):
            if depth == max_depth:
                _put(f"{prefix}{{...}}{suffix}", depth=depth)
            else:
                _put(f"{prefix}{{", depth=depth)
                for key, value in tree.items():
                    _walk(value, prefix=f"{key!r}: ", suffix=",", depth=depth+1)
                _put(f"}}{suffix}", depth=depth)
        elif isinstance(tree, np.ndarray):
            dtype = tree.dtype.name
            shape = str(tree.shape).strip("()").replace(",","")
            _put(f"{prefix}np.{dtype}[{shape}]{suffix}", depth=depth)
        elif isinstance(tree, jnp.ndarray):
            dtype = tree.dtype.name
            shape = str(tree.shape).strip("(,)").replace(" ","")
            _put(f"{prefix}jnp.{dtype}[{shape}]{suffix}", depth=depth)
        elif callable(tree):
            _put(f"{prefix}<fn:{tree.__name__}>{suffix}", depth=depth)
        elif isinstance(tree, (bool, int, float, str)):
            _put(f"{prefix}{type(tree).__name__}({tree!r}){suffix}", depth=depth)
        elif tree is None:
            _put(f"{prefix}None{suffix}", depth=depth)
        else:
            _put(f"{prefix}UNKNOWN_LEAF:{type(tree)}{suffix}", depth=depth)
    _walk(tree, prefix="", suffix="", depth=0)
    return "\n".join(lines)


def format(tree, format_spec: str) -> str:
    """
    A version of `to_str` for use with format strings. `format_spec` should be
    a string in one of the following formats:

    * "{max_depth:d}"
    * "{max_depth:d}.{indent_length:d}"

    Where the `indent` parameter to `to_str` becomes `" "*indent_length`
    (default "  ").
    """
    # parse format spec
    try:
        if not format_spec:
            max_depth = None
            indent_size = 2
        elif '.' in format_spec:
            max_depth_str, indent_size_str = format_spec.split('.')
            max_depth = int(max_depth_str) if max_depth_str else None
            indent_size = int(indent_size_str) if indent_size_str else 2
        else:
            max_depth = int(format_spec)
            indent_size = 2
    except:
        print(f"Invalid format specifier for struct: {format_spec}")
    # render tree
    return to_str(
        tree,
        indent=" " * indent_size,
        max_depth=max_depth,
    )


def size(tree) -> int:
    """Calculates the total number of parameters in the PyTree."""
    return sum(jnp.size(x) for x in jax.tree.leaves(tree))
