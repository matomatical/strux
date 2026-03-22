import dataclasses
import functools
import os
import typing
import warnings

import jax
import jax.numpy as jnp
import numpy as np

try:
    from safetensors import numpy as safetensors_numpy
except ImportError:
    safetensors_numpy = None


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
    if "shape" not in fields:
        Dataclass.shape = property(tree_shape)
    else:
        warnings.warn(
            f"{Class.__name__} has a field named 'shape', so the "
            f"convenience property .shape will not be available",
        )
    Dataclass.__getitem__ = tree_getitem
    if "save" not in fields:
        def _save_method(self, path, *, format=None):
            """Save this struct to disk. See strux.save for details."""
            return save(path, self, format=format)
        Dataclass.save = _save_method
    else:
        warnings.warn(
            f"{Class.__name__} has a field named 'save', so the "
            f"convenience method .save() will not be available",
        )
    if "restore" not in fields:
        def _restore_method(self, path, *, format=None):
            """Load from disk using this struct as the template. See strux.load."""
            return load(path, template=self, format=format)
        Dataclass.restore = _restore_method
    else:
        warnings.warn(
            f"{Class.__name__} has a field named 'restore', so the "
            f"convenience method .restore() will not be available",
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


def tree_shape(tree) -> tuple[int, ...]:
    """
    Return the batch shape of a struct, i.e. the leading dimensions beyond
    each field's base annotation.

    Uses type hints to determine how many trailing dimensions belong to each
    field's base type, and returns the remaining leading (batch) dimensions.
    All data fields must agree on the batch shape.
    """
    cls = type(tree)
    hints = typing.get_type_hints(cls, include_extras=True)
    batch_shape = None
    for name in cls._data_fields:
        hint = hints[name]
        val = getattr(tree, name)
        is_jaxtype = (
            isinstance(hint, type)
            and hasattr(hint, 'dtype')
            and hasattr(hint, 'array_type')
            and hasattr(hint, 'dim_str')
        )
        is_struct = getattr(hint, '_is_strux_struct', False)
        if is_jaxtype:
            base_ndim = len(hint.dim_str.split()) if hint.dim_str else 0
            bs = val.shape[:val.ndim - base_ndim] if base_ndim > 0 else val.shape
        elif is_struct:
            bs = val.shape
        else:
            continue
        if batch_shape is None:
            batch_shape = bs
        elif bs != batch_shape:
            raise ValueError(
                f"Inconsistent batch shapes in {cls.__name__}: "
                f"field '{name}' has batch shape {bs}, expected {batch_shape}"
            )
    return batch_shape if batch_shape is not None else ()


def tree_getitem(tree, index):
    """Index into the batch dimensions of a struct."""
    return jax.tree.map(lambda x: x[index], tree)


def tree_size(tree) -> int:
    """Calculates the total number of parameters in the PyTree."""
    return sum(jnp.size(x) for x in jax.tree.leaves(tree))


# # #
# Serialisation


def _keypath_to_str(keypath) -> str:
    """
    Convert a JAX key path to a '/'-separated string like 'env/hero_pos'.

    Field names (GetAttrKey) are included bare. Dict keys and sequence
    indices are included via repr, so that e.g. string dict keys get
    quoted ('my_key') and remain unambiguous with field names or the
    '/' separator.
    """
    parts = []
    for key in keypath:
        if hasattr(key, 'name'):       # GetAttrKey (dataclass field)
            parts.append(key.name)
        elif hasattr(key, 'key'):       # DictKey
            parts.append(repr(key.key))
        elif hasattr(key, 'idx'):       # SequenceKey (list/tuple)
            parts.append(repr(key.idx))
        else:
            raise TypeError(
                f"Unsupported key type {type(key).__name__} in path"
            )
    return "/".join(parts)


def to_dict(tree) -> dict[str, np.ndarray]:
    """
    Flatten a struct into a dict mapping path strings to numpy arrays.

    Keys are '/'-separated field paths (e.g. 'env/hero_pos', 'score').
    Only data fields (pytree leaves) are included; static/meta fields are not.

    The resulting dict is suitable for saving with `np.savez` or
    `safetensors.numpy.save_file`.
    """
    paths_and_leaves, _ = jax.tree.flatten_with_path(tree)
    d = {}
    for path, leaf in paths_and_leaves:
        key = _keypath_to_str(path)
        if key in d:
            raise ValueError(
                f"Key clash in to_dict: {key!r} appears more than once"
            )
        d[key] = np.asarray(leaf)
    return d


def from_dict(d: dict, *, template):
    """
    Reconstruct a struct from a dict of arrays, using a template for structure.

    The template determines the pytree structure, field order, and static field
    values. Only the data (array) leaves are replaced with values from `d`.

    The keys in `d` must exactly match the keys expected by the template.
    Raises KeyError on missing or extra keys.
    """
    paths_and_leaves, treedef = jax.tree.flatten_with_path(template)
    keys = set(_keypath_to_str(path) for path, _ in paths_and_leaves)
    missing = keys - d.keys()
    extra = d.keys() - keys
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing keys: {sorted(missing)}")
        if extra:
            parts.append(f"extra keys: {sorted(extra)}")
        raise KeyError(f"Key mismatch in from_dict: {'; '.join(parts)}")
    leaves = [jnp.asarray(d[_keypath_to_str(path)]) for path, _ in paths_and_leaves]
    return jax.tree.unflatten(treedef, leaves)


_FORMAT_EXTENSIONS = {
    ".npz": "savez_compressed",
    ".safetensors": "safetensors",
}

_SAVE_FORMATS = {"savez", "savez_compressed", "safetensors"}

_LOAD_FORMATS = {"savez", "savez_compressed", "safetensors"}


def _require_safetensors():
    if safetensors_numpy is None:
        raise ImportError(
            "safetensors is required for the 'safetensors' format; "
            "install it with: pip install strux[safetensors]"
        )


def _infer_format(path):
    ext = os.path.splitext(path)[1]
    if ext not in _FORMAT_EXTENSIONS:
        supported = ", ".join(_FORMAT_EXTENSIONS.keys())
        raise ValueError(
            f"Cannot infer format from extension {ext!r}; "
            f"supported extensions: {supported}. "
            f"Pass format= explicitly to override."
        )
    return _FORMAT_EXTENSIONS[ext]


def save(path, tree, *, format=None):
    """
    Save a struct to disk.

    Format is inferred from the file extension: '.npz' defaults to
    'savez_compressed', '.safetensors' uses safetensors. To save
    uncompressed npz, pass format='savez' explicitly.

    Supported formats:

    * 'savez_compressed' --- compressed numpy npz (default for .npz).
    * 'savez' --- uncompressed numpy npz.
    * 'safetensors' --- safetensors format (requires `safetensors`
      package: `pip install strux[safetensors]`).
    """
    if format is not None:
        fmt = format
    else:
        fmt = _infer_format(path)
    if fmt not in _SAVE_FORMATS:
        raise ValueError(f"Unknown format: {fmt!r}")
    d = to_dict(tree)
    if fmt == "savez_compressed":
        np.savez_compressed(path, **d)
    elif fmt == "savez":
        np.savez(path, **d)
    elif fmt == "safetensors":
        _require_safetensors()
        safetensors_numpy.save_file(d, path)


def load(path, *, template, format=None):
    """
    Load a struct from disk, using a template for the pytree structure.

    The template determines the struct type, field order, and static field
    values. Only the data (array) leaves are loaded from the file.

    Format is inferred from the file extension: '.npz' for numpy npz
    (handles both compressed and uncompressed), '.safetensors' for
    safetensors. Can be specified explicitly via the `format` keyword
    argument.
    """
    if format is not None:
        fmt = format
    else:
        fmt = _infer_format(path)
    if fmt not in _LOAD_FORMATS:
        raise ValueError(f"Unknown format: {fmt!r}")
    if fmt in ("savez", "savez_compressed"):
        d = dict(np.load(path))
    elif fmt == "safetensors":
        _require_safetensors()
        d = safetensors_numpy.load_file(path)
    return from_dict(d, template=template)
