"""
Serialisation: flattening structs to/from dicts of named arrays, and saving
those dicts to disk (npz or safetensors) with atomic writes.
"""

import ast
import dataclasses
import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from strux.schema import (
    schema,
    _ArraySpec,
    _PyScalarSpec,
    _NoneSpec,
    _ClassSpec,
    _ContainerSpec,
    _UnionSpec,
)

# optional safetensors[numpy]
try:
    from safetensors import numpy as safetensors_numpy
except ImportError:
    @dataclasses.dataclass
    class _MissingDependency:
        message: str
        def __getattr__(self, name: str):
            raise ImportError(self.message)

    safetensors_numpy = _MissingDependency(
        "missing optional dependency group strux[safetensors]"
    )


# # #
# Flattening


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


def from_dict(d: dict, *, template, statics=None):
    """
    Reconstruct a struct from a dict of arrays, using a template for structure.

    The template is either an *instance* (the classic path: it determines
    the pytree structure, field order, and static field values, and only
    the data leaves are replaced from `d`) or a struct *class* (the
    template-free path: the structure is derived from the class's schema
    and the dict's keys). In the class case, static fields take their
    values from `statics` — a dict mapping '/'-joined field paths (e.g.
    "activate", "linear1/activate") to values — or from their defaults;
    a missing static value raises KeyError naming the path to pass.

    The template-free path covers schemas made of arrays, scalars, nested
    dataclass structs, containers, and optionals (a union whose data is
    absent restores as None). The saved file records arrays only, never
    type identity, so fields whose type must be *inferred* are restored
    only when the inference is unambiguous: a union arm is chosen only if
    it is the unique arm explaining the saved keys (arms with identical
    key layouts raise), and polymorphic fields (annotated with a base
    class but holding subclass instances) are refused outright — restore
    either with an instance template.

    The keys in `d` must exactly match the keys expected by the template.
    Raises KeyError on missing or extra keys.
    """
    if isinstance(template, type):
        built, consumed = _build_from_dict(template, d, statics or {}, "")
        extra = d.keys() - consumed
        if extra:
            raise KeyError(
                f"Key mismatch in from_dict: extra keys: {sorted(extra)}"
            )
        return built
    if statics is not None:
        raise TypeError(
            "statics= applies only when template is a class; an instance "
            "template already carries its static field values"
        )
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


def _build_from_dict(cls, d, statics, prefix):
    """
    Build an instance of a struct class from saved leaves, guided by its
    schema. Returns (instance, set of consumed keys).
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(
            f"template class {cls.__name__} is not a struct/dataclass"
        )
    field_specs = schema(cls).fields
    values = {}
    consumed = set()
    for field in dataclasses.fields(cls):
        name = field.name
        path = f"{prefix}{name}"
        if name in field_specs:
            value, keys = _build_value(field_specs[name], d, statics, path)
            values[name] = value
            consumed |= keys
        else:
            # static field: from statics, or the field's default
            if path in statics:
                values[name] = statics[path]
            elif field.default is not dataclasses.MISSING:
                values[name] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                values[name] = field.default_factory()
            else:
                raise KeyError(
                    f"static field {path!r} needs a value: pass "
                    f"statics={{{path!r}: ...}}"
                )
    return cls(**values), consumed


def _build_value(spec, d, statics, path):
    if isinstance(spec, (_ArraySpec, _PyScalarSpec)):
        if path not in d:
            raise KeyError(f"missing saved array for {path!r}")
        return jnp.asarray(d[path]), {path}
    elif isinstance(spec, _NoneSpec):
        return None, set()
    elif isinstance(spec, _UnionSpec):
        # an array saved at exactly this path: the array arm
        if path in d:
            return jnp.asarray(d[path]), {path}
        # otherwise the arm must be inferred from the saved keys under
        # this path — and the file records arrays only, never which arm
        # produced them, so the inference is only sound if exactly one
        # arm explains them. A candidate arm must reconstruct AND consume
        # every key under the path (arms with identical key layouts are
        # indistinguishable in the file: refuse rather than guess).
        keys_below = {key for key in d if key.startswith(path + "/")}
        candidates = []
        failures = []
        for arm in spec.arms:
            if isinstance(arm, (_ClassSpec, _ContainerSpec)):
                try:
                    value, consumed = _build_value(arm, d, statics, path)
                except (KeyError, TypeError) as e:
                    failures.append(f"as {arm}: {e}")
                    continue
                if consumed == keys_below:
                    candidates.append((str(arm), value, consumed))
                else:
                    failures.append(
                        f"as {arm}: does not explain saved keys "
                        f"{sorted(keys_below - consumed)}"
                    )
        if not keys_below and any(
            isinstance(arm, _NoneSpec) for arm in spec.arms
        ):
            candidates.append(("None", None, set()))
        if len(candidates) > 1:
            arm_names = ", ".join(name for name, _, _ in candidates)
            raise KeyError(
                f"cannot reconstruct {path!r}: the saved arrays are "
                f"consistent with more than one arm of {spec} "
                f"({arm_names}); the file does not record which arm was "
                "saved — restore with an instance template"
            )
        if candidates:
            (_, value, consumed), = candidates
            return value, consumed
        if failures:
            raise KeyError(
                f"cannot reconstruct {path!r} under any arm of {spec}: "
                + "; ".join(failures)
            )
        raise KeyError(f"missing saved data for {path!r}")
    elif isinstance(spec, _ClassSpec):
        if dataclasses.is_dataclass(spec.cls):
            return _build_from_dict(spec.cls, d, statics, f"{path}/")
        raise TypeError(
            f"{path!r}: cannot reconstruct a {spec.cls.__name__} from a "
            "class template (the saved value's concrete type is unknown); "
            "restore with an instance template"
        )
    elif isinstance(spec, _ContainerSpec):
        # immediate child segments below this path, in saved order
        # (note: quoted dict keys containing '/' are not supported here)
        child_prefix = path + "/"
        segments = []
        for key in d:
            if key.startswith(child_prefix):
                segment = key[len(child_prefix):].split("/", 1)[0]
                if segment not in segments:
                    segments.append(segment)
        consumed = set()
        if spec.kind == "dict":
            out = {}
            for segment in segments:
                dict_key = _ast_literal(segment, path)
                value, keys = _build_value(
                    spec.elems[0], d, statics, f"{path}/{segment}",
                )
                out[dict_key] = value
                consumed |= keys
            return out, consumed
        indices = sorted(int(segment) for segment in segments)
        if indices != list(range(len(indices))):
            raise KeyError(
                f"{path!r}: saved indices {indices} are not contiguous"
            )
        if spec.kind == "tuple":
            if len(indices) != len(spec.elems):
                raise KeyError(
                    f"{path!r}: expected {len(spec.elems)} saved elements "
                    f"({spec}), found {len(indices)}"
                )
            elem_specs = spec.elems
        else:
            elem_specs = (spec.elems[0],) * len(indices)
        items = []
        for i, elem_spec in zip(indices, elem_specs):
            value, keys = _build_value(elem_spec, d, statics, f"{path}/{i}")
            items.append(value)
            consumed |= keys
        if spec.kind == "list":
            return items, consumed
        return tuple(items), consumed
    else:
        raise AssertionError(f"unknown spec {spec!r}")


def _ast_literal(segment, path):
    try:
        return ast.literal_eval(segment)
    except (ValueError, SyntaxError):
        raise KeyError(
            f"{path!r}: cannot parse saved dict key segment {segment!r}"
        )


# # #
# Saving and loading


_FORMAT_EXTENSIONS = {
    ".npz": "savez_compressed",
    ".safetensors": "safetensors",
}

_SAVE_FORMATS = {"savez", "savez_compressed", "safetensors"}

_LOAD_FORMATS = {"savez", "savez_compressed", "safetensors"}


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


def save(path, tree, *, fmt=None, overwrite=False):
    """
    Save a struct to disk.

    Format is inferred from the file extension: '.npz' defaults to
    'savez_compressed', '.safetensors' uses safetensors. To save
    uncompressed npz, pass fmt='savez' explicitly.

    Supported formats:

    * 'savez_compressed' --- compressed numpy npz (default for .npz).
    * 'savez' --- uncompressed numpy npz.
    * 'safetensors' --- safetensors format (requires `safetensors`
      package: `pip install strux[safetensors]`).

    By default, raises FileExistsError if the destination file already
    exists, to prevent accidental data loss; pass overwrite=True to
    replace it (e.g. for repeatedly saving the latest checkpoint during
    training).

    The write is atomic: data is first written to a temporary file in the
    same directory, which is then renamed over the destination, so an
    interrupted save never leaves a partial file at the destination. (The
    overwrite=False existence check itself is not atomic, so simultaneous
    savers to the same path can still race it; use distinct paths per
    process if saving concurrently.)

    Note: for the npz formats, '.npz' is appended to the path if it
    doesn't already end in '.npz' (matching numpy's savez behaviour).
    Safetensors writes to the exact path given. For consistent behaviour,
    use '.npz' or '.safetensors' extensions explicitly.
    """
    if fmt is None:
        fmt = _infer_format(path)
    if fmt not in _SAVE_FORMATS:
        raise ValueError(f"Unknown format: {fmt!r}")
    # resolve the true destination (numpy conventionally appends .npz)
    dest = os.fspath(path)
    if fmt in ("savez", "savez_compressed") and not dest.endswith(".npz"):
        dest = dest + ".npz"
    # check for existing file to prevent silent data loss
    if not overwrite and os.path.exists(dest):
        raise FileExistsError(
            f"File already exists: {dest!r}. "
            f"Pass overwrite=True to replace it."
        )
    d = to_dict(tree)
    # write to a temporary file in the destination directory, then rename it
    # over the destination, so an interrupted save can't corrupt an existing
    # file
    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(dest) or ".",
        prefix=os.path.basename(dest) + ".tmp.",
    )
    try:
        if fmt == "savez_compressed":
            with os.fdopen(fd, "wb") as f:
                np.savez_compressed(f, **d)
        elif fmt == "savez":
            with os.fdopen(fd, "wb") as f:
                np.savez(f, **d)
        elif fmt == "safetensors":
            os.close(fd)
            safetensors_numpy.save_file(d, tmp)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load(path, *, template, fmt=None, statics=None):
    """
    Load a struct from disk, using a template for the pytree structure.

    The template is either an instance (which determines the struct type,
    field order, and static field values; only the data leaves are loaded
    from the file) or a struct class (template-free restore: the structure
    is derived from the class's schema and the saved keys, and static
    fields come from `statics` or their defaults — see `strux.from_dict`).

    Format is inferred from the file extension: '.npz' for numpy npz
    (handles both compressed and uncompressed), '.safetensors' for
    safetensors. Can be specified explicitly via the `fmt` keyword
    argument.
    """
    if fmt is None:
        fmt = _infer_format(path)
    if fmt not in _LOAD_FORMATS:
        raise ValueError(f"Unknown format: {fmt!r}")
    if fmt in ("savez", "savez_compressed"):
        d = dict(np.load(path))
    elif fmt == "safetensors":
        d = safetensors_numpy.load_file(path)
    return from_dict(d, template=template, statics=statics)
