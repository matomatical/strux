"""
Serialisation: flattening structs to/from dicts of named arrays, and saving
those dicts to disk (npz or safetensors) with atomic writes.

Alongside the arrays, saved files carry a small string-to-string metadata
mapping (the safetensors metadata header, or a reserved npz entry) recording
what the arrays alone cannot: literal static field values, which union arm
or subclass was saved at each path, and true dtypes where the container
format is lossy (npz stores ml_dtypes like bfloat16 as raw bytes). Restore
uses the metadata to rebuild structure without guessing; files without
metadata (saved by earlier strux versions, or foreign files) restore from
the arrays alone, refusing wherever the structure would be ambiguous.
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
    SchemaError,
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
    from safetensors import safe_open
except ImportError:
    @dataclasses.dataclass
    class _MissingDependency:
        message: str
        def __getattr__(self, name: str):
            raise ImportError(self.message)
        def __call__(self, *args, **kwargs):
            raise ImportError(self.message)

    safetensors_numpy = _MissingDependency(
        "missing optional dependency group strux[safetensors]"
    )
    safe_open = safetensors_numpy


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


# # #
# Metadata
#
# A flat str -> str mapping (the lowest common denominator: safetensors
# metadata headers are exactly this). Entries:
#
#   "strux": format version ("2")
#   "class <path>": module-qualified name of the dataclass at <path>
#                   (recorded at every dataclass node; "class" alone for
#                   the root)
#   "arm <path>":   which arm of a union-annotated field was saved:
#                   "class" | "dict" | "list" | "tuple" | "tuple_variadic"
#                   | "none" | "array" | "scalar"
#   "static <path>": repr of a literal static field value (restored with
#                   ast.literal_eval — literal parsing only, never code
#                   execution)
#   "dtype <path>": true dtype name of a leaf the container format stores
#                   as raw bytes (ml_dtypes under npz); restored by view-
#                   casting
#
# Paths never contain spaces before the first '/', so splitting a key on
# its first space is unambiguous.


_METADATA_VERSION = "2"
_NPZ_METADATA_KEY = "__strux__"


def _meta_key(kind, path):
    return f"{kind} {path}" if path else kind


def _fullname(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


def metadata(tree) -> dict[str, str]:
    """
    The metadata mapping `strux.save` records alongside a tree's arrays
    (see the module docstring): format version, class and union-arm tags,
    literal static values, and true dtype names for leaves whose npz
    storage is lossy. The companion of `strux.to_dict`: pass the result to
    `strux.from_dict` as `meta=` to restore with the same structural
    guidance a saved file provides.
    """
    return _collect_metadata(tree, to_dict(tree), record_dtypes=True)


def _collect_metadata(tree, d, record_dtypes):
    """
    Build the metadata mapping for a tree about to be saved. `d` is the
    tree's to_dict flattening (consulted for lossy-dtype records).
    """
    meta = {"strux": _METADATA_VERSION}
    if record_dtypes:
        for key, arr in d.items():
            dtype = np.dtype(arr.dtype)
            if dtype.kind == "V" and dtype.names is None:
                meta[_meta_key("dtype", key)] = dtype.name
    if dataclasses.is_dataclass(tree) and not isinstance(tree, type):
        try:
            _collect_class_meta(tree, "", meta)
        except SchemaError:
            # trees whose annotations don't compile to a schema still save
            # (arrays only); they restore with an instance template
            pass
    return meta


def _collect_class_meta(obj, path, meta):
    """
    Record a dataclass node's identity, its literal statics, and structure
    tags beneath it. Walks the *value's* dynamic type, mirroring how
    to_dict flattens (a subclass instance flattens with its own fields).
    """
    cls = type(obj)
    meta[_meta_key("class", path)] = _fullname(cls)
    field_specs = schema(cls).fields
    for field in dataclasses.fields(cls):
        field_path = f"{path}/{field.name}" if path else field.name
        value = getattr(obj, field.name)
        if field.name in field_specs:
            _collect_value_meta(value, field_specs[field.name], field_path, meta)
        else:
            literal = _as_literal(value)
            if literal is not None:
                meta[_meta_key("static", field_path)] = literal


def _collect_value_meta(value, spec, path, meta):
    if isinstance(spec, _UnionSpec):
        arm = _match_arm(value, spec)
        if arm is None:
            return  # validated values always match an arm; stay silent
        meta[_meta_key("arm", path)] = _arm_token(arm)
        _collect_value_meta(value, arm, path, meta)
    elif isinstance(spec, _ClassSpec):
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            _collect_class_meta(value, path, meta)
    elif isinstance(spec, _ContainerSpec):
        if spec.kind == "dict":
            for key, item in value.items():
                _collect_value_meta(
                    item, spec.elems[0], f"{path}/{key!r}", meta,
                )
        elif spec.kind == "tuple":
            for i, (item, elem_spec) in enumerate(zip(value, spec.elems)):
                _collect_value_meta(item, elem_spec, f"{path}/{i}", meta)
        else:   # list, tuple_variadic
            for i, item in enumerate(value):
                _collect_value_meta(item, spec.elems[0], f"{path}/{i}", meta)
    # array, scalar, and None leaves carry no structure


def _match_arm(value, spec):
    """The arm of a union spec that this value was validated under."""
    if value is None:
        return next(
            (arm for arm in spec.arms if isinstance(arm, _NoneSpec)), None,
        )
    class_arms = [
        arm for arm in spec.arms
        if isinstance(arm, _ClassSpec) and isinstance(value, arm.cls)
    ]
    if class_arms:
        # most-derived arm class wins (e.g. Base | Sub holding a Sub)
        best = class_arms[0]
        for arm in class_arms[1:]:
            if issubclass(arm.cls, best.cls):
                best = arm
        return best
    if isinstance(value, dict):
        kinds = ("dict",)
    elif isinstance(value, list):
        kinds = ("list",)
    elif isinstance(value, tuple):
        kinds = ("tuple", "tuple_variadic")
    else:
        scalar_arm = next(
            (arm for arm in spec.arms if isinstance(arm, _PyScalarSpec)),
            None,
        )
        array_arm = next(
            (arm for arm in spec.arms if isinstance(arm, _ArraySpec)), None,
        )
        if isinstance(value, (bool, int, float, complex)) and scalar_arm:
            return scalar_arm
        return array_arm or scalar_arm
    return next(
        (
            arm for arm in spec.arms
            if isinstance(arm, _ContainerSpec) and arm.kind in kinds
        ),
        None,
    )


def _arm_token(arm):
    if isinstance(arm, _ClassSpec):
        return "class"
    if isinstance(arm, _ContainerSpec):
        return arm.kind
    if isinstance(arm, _NoneSpec):
        return "none"
    if isinstance(arm, _PyScalarSpec):
        return "scalar"
    return "array"


def _as_literal(value):
    """
    The repr of a value, if parsing it back with ast.literal_eval yields an
    equal value of the same type; None otherwise (the value is not
    literal-serialisable).
    """
    try:
        rep = repr(value)
        parsed = ast.literal_eval(rep)
    except Exception:
        return None
    try:
        if type(parsed) is type(value) and bool(parsed == value):
            return rep
    except Exception:
        return None
    return None


# # #
# Class resolution
#
# Recorded class names resolve against classes Python has already imported
# (the template class and its transitive subclasses) — a name lookup, never
# an import and never code execution, keeping npz/safetensors' security
# level.


def _find_subclass(base, fullname):
    seen = set()
    stack = [base]
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        if _fullname(cls) == fullname:
            return cls
        stack.extend(cls.__subclasses__())
    return None


def _resolve_class(base, meta, path):
    """
    The class to rebuild at this path: the recorded class if metadata names
    an (imported) subclass of `base`, otherwise `base` itself.
    """
    recorded = meta.get(_meta_key("class", path))
    if recorded is None or recorded == _fullname(base):
        return base
    found = _find_subclass(base, recorded)
    if found is None:
        raise TypeError(
            f"{path or 'the root'}: the checkpoint records class "
            f"{recorded!r}, which is not {_fullname(base)} or any imported "
            "subclass of it; import the module defining that class, or "
            "restore with an instance template"
        )
    return found


def _narrow_class_arms(arms, recorded):
    """
    Narrow a union's class arms using the recorded class name: an exact
    arm match wins, else arms whose (imported) subclasses include the
    recorded class, most-derived first.
    """
    if recorded is None:
        return arms
    exact = [arm for arm in arms if _fullname(arm.cls) == recorded]
    if exact:
        return exact
    resolving = [
        arm for arm in arms if _find_subclass(arm.cls, recorded) is not None
    ]
    for arm in resolving:
        if all(issubclass(arm.cls, other.cls) for other in resolving):
            return [arm]
    return resolving


# # #
# Reconstruction


def from_dict(d: dict, *, template, statics=None, meta=None):
    """
    Reconstruct a struct from a dict of arrays, using a template for structure.

    The template is either an *instance* (the classic path: it determines
    the pytree structure, field order, and static field values, and only
    the data leaves are replaced from `d`) or a struct *class* (the
    template-free path: the structure is derived from the class's schema,
    the dict's keys, and the metadata). In the class case, static fields
    take their values from `statics` — a dict mapping '/'-joined field
    paths (e.g. "activate", "linear1/activate") to values — then from the
    metadata's recorded literals, then from their defaults; a static
    resolvable by none of these raises KeyError naming the path to pass.

    `meta` is the str -> str metadata mapping written at save time (strux.load
    passes it through automatically). It records which union arm or subclass
    was saved at each path, literal static values, and true dtypes for
    leaves the file stores as raw bytes. Without it, the template-free path
    covers schemas made of arrays, scalars, nested dataclass structs,
    containers, and optionals (a union whose data is absent restores as
    None); fields whose type must be *inferred* are restored only when the
    inference is unambiguous: a union arm is chosen only if it is the
    unique arm explaining the saved keys (arms with identical key layouts
    raise), and polymorphic fields (annotated with a base class but holding
    subclass instances) are refused — restore those with an instance
    template.

    Restore is strict: the keys in `d` must exactly match the keys expected
    by the template (KeyError on missing or extra keys), and with an
    instance template every saved leaf must match the template leaf's shape
    and dtype (ValueError listing every mismatch) and the recorded
    structure tags must agree with the template's structure (TypeError).
    """
    meta = dict(meta) if meta else {}
    # view raw-bytes leaves back to their recorded dtypes (npz cannot
    # record ml_dtypes like bfloat16 natively)
    dtype_records = {
        key: meta[_meta_key("dtype", key)]
        for key in d
        if _meta_key("dtype", key) in meta
    }
    if dtype_records:
        d = dict(d)
        for key, name in dtype_records.items():
            d[key] = np.asarray(d[key]).view(np.dtype(name))
    if isinstance(template, type):
        cls = _resolve_class(template, meta, "")
        built, consumed = _build_from_dict(cls, d, statics or {}, "", meta)
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
    _check_template_structure(template, meta)
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
    leaves = []
    mismatches = []
    for path, leaf in paths_and_leaves:
        key = _keypath_to_str(path)
        saved = d[key]
        if isinstance(leaf, (bool, int, float, complex)):
            if tuple(saved.shape) == ():
                leaves.append(type(leaf)(saved.item()))
            else:
                mismatches.append(
                    f"{key!r}: template has a python scalar "
                    f"({type(leaf).__name__}), saved shape is "
                    f"{tuple(saved.shape)}"
                )
                leaves.append(leaf)
        elif hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            if (
                tuple(saved.shape) != tuple(leaf.shape)
                or np.dtype(saved.dtype) != np.dtype(leaf.dtype)
            ):
                mismatches.append(
                    f"{key!r}: template expects "
                    f"{np.dtype(leaf.dtype).name}{list(leaf.shape)}, saved "
                    f"is {np.dtype(saved.dtype).name}{list(saved.shape)}"
                )
                leaves.append(leaf)
            else:
                leaves.append(jnp.asarray(saved))
        else:
            leaves.append(jnp.asarray(saved))
    if mismatches:
        raise ValueError(
            "cannot restore: saved leaves do not match the template "
            "(strict restore refuses shape/dtype mismatches):\n  "
            + "\n  ".join(mismatches)
        )
    return jax.tree.unflatten(treedef, leaves)


def _check_template_structure(template, meta):
    """
    Cross-check an instance template's structure against the checkpoint's
    recorded tags: a checkpoint saved with one union arm or class must not
    silently restore into a template carrying another, even when the key
    layouts happen to coincide.
    """
    if not meta:
        return
    if not dataclasses.is_dataclass(template) or isinstance(template, type):
        return
    expected = {}
    try:
        _collect_class_meta(template, "", expected)
    except SchemaError:
        return
    conflicts = []
    for key, value in expected.items():
        kind, _, path = key.partition(" ")
        if kind in ("arm", "class") and key in meta and meta[key] != value:
            conflicts.append(
                f"at {path or '(root)'}: checkpoint has {meta[key]!r}, "
                f"template has {value!r}"
            )
    if conflicts:
        raise TypeError(
            "cannot restore: the checkpoint's recorded structure disagrees "
            "with the instance template:\n  " + "\n  ".join(conflicts)
        )


def _build_from_dict(cls, d, statics, prefix, meta):
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
            value, keys = _build_value(field_specs[name], d, statics, path, meta)
            values[name] = value
            consumed |= keys
        else:
            # static field: from statics, else recorded literal, else the
            # field's default
            literal = meta.get(_meta_key("static", path))
            if path in statics:
                values[name] = statics[path]
            elif literal is not None:
                values[name] = ast.literal_eval(literal)
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


def _build_value(spec, d, statics, path, meta):
    if isinstance(spec, _ArraySpec):
        if path not in d:
            raise KeyError(f"missing saved array for {path!r}")
        return jnp.asarray(d[path]), {path}
    elif isinstance(spec, _PyScalarSpec):
        if path not in d:
            raise KeyError(f"missing saved array for {path!r}")
        return spec.cls(np.asarray(d[path]).item()), {path}
    elif isinstance(spec, _NoneSpec):
        return None, set()
    elif isinstance(spec, _UnionSpec):
        token = meta.get(_meta_key("arm", path))
        if token is None:
            return _infer_union_arm(spec, spec.arms, d, statics, path, meta)
        # the arm is recorded: rebuild under it, no inference needed
        if token == "none":
            return None, set()
        matching = [arm for arm in spec.arms if _arm_token(arm) == token]
        if token == "class":
            matching = _narrow_class_arms(
                matching, meta.get(_meta_key("class", path)),
            )
        if not matching:
            raise TypeError(
                f"{path!r}: the checkpoint records a {token!r} arm, but "
                f"{spec} has no matching arm; the class definition may "
                "have changed since saving"
            )
        if len(matching) == 1:
            return _build_value(matching[0], d, statics, path, meta)
        # several arms of the recorded kind: fall back to inference among
        # them (refusing if more than one explains the saved keys)
        return _infer_union_arm(spec, tuple(matching), d, statics, path, meta)
    elif isinstance(spec, _ClassSpec):
        cls = _resolve_class(spec.cls, meta, path)
        if dataclasses.is_dataclass(cls):
            return _build_from_dict(cls, d, statics, f"{path}/", meta)
        raise TypeError(
            f"{path!r}: cannot reconstruct a {cls.__name__} from a "
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
                    spec.elems[0], d, statics, f"{path}/{segment}", meta,
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
            value, keys = _build_value(elem_spec, d, statics, f"{path}/{i}", meta)
            items.append(value)
            consumed |= keys
        if spec.kind == "list":
            return items, consumed
        return tuple(items), consumed
    else:
        raise AssertionError(f"unknown spec {spec!r}")


def _infer_union_arm(spec, arms, d, statics, path, meta):
    """
    Choose a union arm from the saved keys alone (no arm recorded, or
    several arms of the recorded kind). The file records arrays by path,
    never which arm produced them, so the inference is only sound if
    exactly one arm explains them.
    """
    # an array saved at exactly this path: the array arm
    if path in d:
        return jnp.asarray(d[path]), {path}
    # otherwise a candidate arm must reconstruct AND consume every key
    # under the path (arms with identical key layouts are indistinguishable
    # in the file: refuse rather than guess)
    keys_below = {key for key in d if key.startswith(path + "/")}
    candidates = []
    failures = []
    for arm in arms:
        if isinstance(arm, (_ClassSpec, _ContainerSpec)):
            try:
                value, consumed = _build_value(arm, d, statics, path, meta)
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
        isinstance(arm, _NoneSpec) for arm in arms
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

    Alongside the arrays, the file records a small metadata mapping (see
    the module docstring): literal static values, union-arm and class tags,
    and true dtypes for leaves npz stores as raw bytes. In npz files the
    metadata occupies a reserved entry, so a tree with a field path equal
    to '__strux__' cannot be saved as npz.

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
    is_npz = fmt in ("savez", "savez_compressed")
    meta = _collect_metadata(tree, d, record_dtypes=is_npz)
    if is_npz and _NPZ_METADATA_KEY in d:
        raise ValueError(
            f"cannot save as npz: the tree has a field path "
            f"{_NPZ_METADATA_KEY!r}, which is reserved for strux metadata"
        )
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
                np.savez_compressed(
                    f, **d, **{_NPZ_METADATA_KEY: np.array(repr(meta))},
                )
        elif fmt == "savez":
            with os.fdopen(fd, "wb") as f:
                np.savez(f, **d, **{_NPZ_METADATA_KEY: np.array(repr(meta))})
        elif fmt == "safetensors":
            os.close(fd)
            safetensors_numpy.save_file(d, tmp, metadata=meta)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_file(path, fmt):
    """
    Read a saved file: (dict of arrays, metadata mapping). The metadata is
    empty for files without any (saved by earlier strux versions, or
    foreign npz/safetensors files).
    """
    if fmt in ("savez", "savez_compressed"):
        d = dict(np.load(path))
        meta = {}
        if _NPZ_METADATA_KEY in d:
            raw = d.pop(_NPZ_METADATA_KEY)
            try:
                meta = ast.literal_eval(raw.item())
            except (ValueError, TypeError, SyntaxError, AttributeError):
                meta = None
            if not isinstance(meta, dict):
                raise ValueError(
                    f"{path}: the reserved entry {_NPZ_METADATA_KEY!r} does "
                    "not hold a strux metadata mapping"
                )
        return d, meta
    elif fmt == "safetensors":
        with safe_open(path, framework="np") as f:
            meta = f.metadata() or {}
            d = {key: f.get_tensor(key) for key in f.keys()}
        return d, meta
    raise ValueError(f"Unknown format: {fmt!r}")


def load(path, *, template, fmt=None, statics=None):
    """
    Load a struct from disk, using a template for the pytree structure.

    The template is either an instance (which determines the struct type,
    field order, and static field values; only the data leaves are loaded
    from the file, and each must match the template leaf's shape and
    dtype) or a struct class (template-free restore: the structure is
    derived from the class's schema, the saved keys, and the saved
    metadata, and static fields come from `statics`, recorded literals, or
    their defaults — see `strux.from_dict`).

    Format is inferred from the file extension: '.npz' for numpy npz
    (handles both compressed and uncompressed), '.safetensors' for
    safetensors. Can be specified explicitly via the `fmt` keyword
    argument.
    """
    if fmt is None:
        fmt = _infer_format(path)
    if fmt not in _LOAD_FORMATS:
        raise ValueError(f"Unknown format: {fmt!r}")
    d, meta = _read_file(path, fmt)
    return from_dict(d, template=template, statics=statics, meta=meta)
