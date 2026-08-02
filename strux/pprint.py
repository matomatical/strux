"""
Pretty printing for pytrees: multi-line renders with shape/dtype summaries
for array leaves. Works on any pytree, not just structs.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np


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
        elif isinstance(tree, tuple) and hasattr(tree, '_fields'):
            # namedtuple
            if depth == max_depth:
                _put(f"{prefix}{type(tree).__name__}(...){suffix}", depth=depth)
            else:
                _put(f"{prefix}{type(tree).__name__}(", depth=depth)
                for field, value in zip(tree._fields, tree):
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
            name = getattr(tree, '__name__', None)
            if name is not None:
                _put(f"{prefix}<fn:{name}>{suffix}", depth=depth)
            else:
                _put(f"{prefix}{repr(tree)}{suffix}", depth=depth)
        elif isinstance(tree, (bool, int, float, complex, str)):
            _put(f"{prefix}{type(tree).__name__}({tree!r}){suffix}", depth=depth)
        elif tree is None:
            _put(f"{prefix}None{suffix}", depth=depth)
        else:
            # registered pytree nodes that aren't dataclasses: render one
            # level via the registry's keyed flattening; anything else is
            # an unknown leaf
            try:
                keys_and_children, _ = (
                    jax.tree_util.default_registry.flatten_one_level_with_keys(
                        tree
                    )
                )
            except (TypeError, ValueError):
                _put(f"{prefix}UNKNOWN_LEAF:{type(tree)}{suffix}", depth=depth)
                return
            if depth == max_depth:
                _put(f"{prefix}{type(tree).__name__}(...){suffix}", depth=depth)
            else:
                _put(f"{prefix}{type(tree).__name__}(", depth=depth)
                for key, child in keys_and_children:
                    key_str = str(key)
                    if key_str.startswith("."):
                        child_prefix = f"{key_str[1:]}="
                    else:
                        child_prefix = f"{key_str}: "
                    _walk(child, prefix=child_prefix, suffix=",", depth=depth+1)
                _put(f"){suffix}", depth=depth)
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
    # empty spec -> delegate to str(self), matching Python's default behaviour
    if not format_spec:
        return str(tree)
    # parse format spec
    try:
        if '.' in format_spec:
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
