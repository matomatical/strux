import dataclasses
import functools
import typing
import warnings

import jax
import jax.numpy as jnp
import numpy as np


def struct(Class=None, *, static_fieldnames: typing.Sequence[str] = ()):
    """
    Transform a class into an immutable dataclass that is also registered as a
    JAX PyTree. Can be used as a bare decorator or with keyword arguments:

    ```
    @strux.struct
    class MyDataClass:
        field1: int
        # etc.

    @strux.struct(static_fieldnames=("label",))
    class MyOtherDataClass:
        field1: int
        label: str
    ```
    """
    if Class is None:
        return functools.partial(struct, static_fieldnames=static_fieldnames)
    # wrap class as an immutable Python dataclass
    Dataclass = dataclasses.dataclass(Class, frozen=True)

    # decide which fields are data vs. static
    fields = [field.name for field in dataclasses.fields(Dataclass)]
    data_fields = [name for name in fields if name not in static_fieldnames]
    meta_fields = [name for name in fields if name in static_fieldnames]
    missing_fields = set(static_fieldnames) - set(meta_fields)
    if missing_fields:
        raise ValueError(f"Invalid static_fieldnames {missing_fields}")
    Dataclass._data_fields = data_fields
    Dataclass._meta_fields = meta_fields
    
    # register dataclass as a JAX pytree node
    jax.tree_util.register_dataclass(
        nodetype=Dataclass,
        data_fields=data_fields,
        meta_fields=meta_fields,
    )
    
    # overwrite string render methods to use pretty printing
    if "__repr__" not in Class.__dict__:
        Dataclass.__repr__ = to_str
    if "__str__" not in Class.__dict__:
        Dataclass.__str__ = to_str
    if "__format__" not in Class.__dict__:
        Dataclass.__format__ = tree_format
    
    # add some other convenience methods
    if "replace" not in fields:
        Dataclass.replace = dataclasses.replace
    else:
        warnings.warn(
            f"{Class.__name__} has a field named 'replace', so the "
            f"convenience method .replace() will not be available",
        )
    if "size" not in fields:
        Dataclass.size = property(tree_size)
    else:
        warnings.warn(
            f"{Class.__name__} has a field named 'size', so the "
            f"convenience property .size will not be available",
        )

    # allow type subscripting for annotating batched/vmapped pytrees,
    Dataclass._is_strux_struct = True
    Dataclass.__class_getitem__ = classmethod(_make_struct_annotation)
    
    # done!
    return Dataclass


@functools.lru_cache(maxsize=None)
def _make_struct_annotation(struct_cls, dims):
    """
    Create a type annotation representing a batched/vmapped struct.

    For example, given:

        @strux.struct
        class Env:
            pos: Int[Array, "2"]
            walls: Bool[Array, "h w"]

    Then Env["batch"] produces a type where isinstance checks verify as if it
    were defined:

        @strux.struct
        class Envs:
            pos: Int[Array, "batch 2"]
            walls: Bool[Array, "batch h w"]
    """
    hints = typing.get_type_hints(struct_cls, include_extras=True)
    expanded = {}
    for name, hint in hints.items():
        # don't propagate dims to meta fields
        if name in struct_cls._meta_fields:
            continue
        # propagate dims to jaxtype and struct fields
        is_jaxtype = (
            isinstance(hint, type)
            and hasattr(hint, 'dtype')
            and hasattr(hint, 'array_type')
            and hasattr(hint, 'dim_str')
        )
        is_struct = getattr(hint, '_is_strux_struct', False)
        if is_jaxtype:
            new_dims = f"{dims} {hint.dim_str}".strip()
            expanded[name] = hint.dtype[hint.array_type, new_dims]
        elif is_struct:
            expanded[name] = hint[dims]
        # unclear how to propagate otherwise
        else:
            _scalar_hints = {
                float: 'Float[Array, ""]',
                int: 'Int[Array, ""]',
                bool: 'Bool[Array, ""]',
                complex: 'Complex[Array, ""]',
            }
            msg = (
                f"Cannot batch data field '{name}' of {struct_cls.__name__}: "
                f"type {hint} is not a jaxtyping annotation or strux struct"
            )
            if hint in _scalar_hints:
                msg += (
                    f". If '{name}' is a scalar array, consider "
                    f"annotating it as {_scalar_hints[hint]} instead of "
                    f"{hint.__name__}"
                )
            raise TypeError(msg)
    return _StructAnnotationMeta(
        f'{struct_cls.__name__}["{dims}"]',
        (),
        {
            '_struct_type': struct_cls,
            '_field_hints': expanded,
        },
    )


class _StructAnnotationMeta(type):
    """Metaclass for batched struct type annotations with isinstance support."""
    def __instancecheck__(cls, instance):
        if not isinstance(instance, cls._struct_type):
            return False
        for field_name, expected_type in cls._field_hints.items():
            if not isinstance(getattr(instance, field_name), expected_type):
                return False
        return True


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
                state = vars(tree)
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
            shape = str(tree.shape).strip("(,)").replace(" ","")
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


def tree_format(tree, format_spec: str) -> str:
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
    except (ValueError, TypeError):
        raise ValueError(f"Invalid format specifier for struct: {format_spec!r}")
    # render tree
    return to_str(
        tree,
        indent=" " * indent_size,
        max_depth=max_depth,
    )


def tree_size(tree) -> int:
    """Calculates the total number of parameters in the PyTree."""
    return sum(jnp.size(x) for x in jax.tree.leaves(tree))
