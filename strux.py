import ast
import collections.abc
import dataclasses
import functools
import inspect
import os
import re
import sys
import tempfile
import types
import typing
import warnings

import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np

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

# dataclass_transform (PEP 681) tells static type checkers like mypy that
# @strux.struct generates dataclass semantics (an __init__ from the field
# annotations, frozen instances). In the standard library from python
# 3.11; harmless no-op fallback below it.
try:
    from typing import dataclass_transform
except ImportError:
    try:
        from typing_extensions import dataclass_transform
    except ImportError:
        def dataclass_transform(**kwargs):
            def identity(cls_or_fn):
                return cls_or_fn
            return identity


def _is_jaxtype(hint) -> bool:
    """
    Is this type hint a jaxtyping array annotation? Jaxtyping is the only
    supported array annotation framework.
    """
    return isinstance(hint, type) and issubclass(hint, jaxtyping.AbstractArray)


# # # 
# Core wrapper


@dataclass_transform(frozen_default=True, field_specifiers=(dataclasses.field,))
def struct(
    Class=None,
    *,
    static_fieldnames: typing.Sequence[str] = (),
    check: bool = True,
):
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

    By default, construction validates the data fields against the schema
    derived from the field annotations (see `strux.schema`): dtype kind and
    trailing (element) dims are enforced, leading batch dims are free but
    must agree across fields. Pass `check=False` to skip this validation
    (e.g. for a class constructed in a measured hot path).
    """
    if Class is None:
        return functools.partial(
            struct,
            static_fieldnames=static_fieldnames,
            check=check,
        )
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
    
    # register as a JAX pytree node. Reconstruction through JAX's tree
    # machinery (tree.map results, vmap/scan/jit outputs, gradients) is
    # *structural*: it bypasses __init__ and hence validation, because
    # transformed trees legitimately carry leaves that differ from the
    # declared element types (tree.map(jnp.array_equal, ...) yields bool
    # scalars, cotangents carry float0, masks change dtypes). Validation
    # guards the user API boundary instead: direct construction and
    # .replace. (For the same reason, __post_init__ logic runs on direct
    # construction only, never on tree reconstruction.)
    data_fields_tuple = tuple(data_fields)
    meta_fields_tuple = tuple(meta_fields)
    def _flatten_with_keys(obj):
        return (
            [
                (jax.tree_util.GetAttrKey(name), getattr(obj, name))
                for name in data_fields_tuple
            ],
            tuple(getattr(obj, name) for name in meta_fields_tuple),
        )
    def _flatten(obj):
        return (
            tuple(getattr(obj, name) for name in data_fields_tuple),
            tuple(getattr(obj, name) for name in meta_fields_tuple),
        )
    def _unflatten(meta, data):
        obj = object.__new__(Dataclass)
        for name, value in zip(data_fields_tuple, data):
            object.__setattr__(obj, name, value)
        for name, value in zip(meta_fields_tuple, meta):
            object.__setattr__(obj, name, value)
        return obj
    jax.tree_util.register_pytree_with_keys(
        Dataclass,
        _flatten_with_keys,
        _unflatten,
        flatten_func=_flatten,
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
            f"convenience method .replace() will not be available; use "
            f"dataclasses.replace(obj, ...) instead",
        )
    if "size" not in fields:
        Dataclass.size = property(tree_size)
    else:
        warnings.warn(
            f"{Class.__name__} has a field named 'size', so the "
            f"convenience property .size will not be available; use "
            f"strux.tree_size(obj) instead",
        )
    if "shape" not in fields:
        Dataclass.shape = property(tree_shape)
    else:
        warnings.warn(
            f"{Class.__name__} has a field named 'shape', so the "
            f"convenience property .shape will not be available; use "
            f"strux.tree_shape(obj) instead",
        )
    Dataclass.__getitem__ = tree_getitem
    if "save" not in fields:
        def _save_method(self, path, *, fmt=None, overwrite=False):
            """Save this struct to disk. See strux.save for details."""
            return save(path, self, fmt=fmt, overwrite=overwrite)
        Dataclass.save = _save_method
    else:
        warnings.warn(
            f"{Class.__name__} has a field named 'save', so the "
            f"convenience method .save() will not be available; use "
            f"strux.save(path, obj) instead",
        )
    if "restore" not in fields:
        def _restore_method(self, path, *, fmt=None):
            """Load from disk using this struct as the template. See strux.load."""
            return load(path, template=self, fmt=fmt)
        Dataclass.restore = _restore_method
    else:
        warnings.warn(
            f"{Class.__name__} has a field named 'restore', so the "
            f"convenience method .restore() will not be available; use "
            f"strux.load(path, template=obj) instead",
        )

    # allow type subscripting for annotating batched/vmapped pytrees,
    Dataclass._is_strux_struct = True
    Dataclass.__class_getitem__ = classmethod(_make_struct_annotation)

    # construction checking: the constructor validates the new instance
    # against the schema derived from the field annotations (dtype kind and
    # element dims enforced, leading batch dims free but consistent across
    # fields; see strux.schema). The runtime __init__ deliberately carries
    # no annotations of its own: external runtime checkers that wrap
    # dataclass constructors (such as jaxtyping's import hook) would
    # otherwise enforce the rank-exact field annotations and reject
    # legitimately batched constructions (JAX rebuilds structs through the
    # constructor when unflattening, so vmap/scan/tree-stack results all
    # construct with batch-dim'd leaves). Static type checkers are
    # unaffected: they derive __init__ from the field annotations (PEP 681).
    original_init = Dataclass.__init__
    if check:
        def checked_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            _validate_struct(self)
        # copy identity but not annotations, and expose an
        # annotation-stripped signature: external checkers must find
        # nothing to enforce, whether they read __annotations__ or
        # inspect.signature (which is also why there is no __wrapped__
        # link back to the annotated original)
        checked_init.__name__ = original_init.__name__
        checked_init.__qualname__ = original_init.__qualname__
        checked_init.__module__ = original_init.__module__
        checked_init.__doc__ = original_init.__doc__
        checked_init.__signature__ = _strip_annotations_signature(
            original_init
        )
        Dataclass.__init__ = checked_init
    else:
        original_init.__annotations__ = {}
        original_init.__signature__ = _strip_annotations_signature(
            original_init
        )

    # done!
    return Dataclass


def _strip_annotations_signature(fn):
    signature = inspect.signature(fn)
    return signature.replace(
        parameters=[
            param.replace(annotation=inspect.Parameter.empty)
            for param in signature.parameters.values()
        ],
        return_annotation=inspect.Signature.empty,
    )


class SchemaError(TypeError):
    """
    A struct's field annotations cannot be compiled into a schema. Raised
    when the schema is first needed (typically at first construction), not
    at class decoration.
    """


class ValidationError(TypeError):
    """
    A value does not satisfy a struct's schema at construction: wrong type
    or dtype kind, mismatched element dims, or inconsistent batch dims
    across fields.
    """


# A struct's field annotations describe one *element*, but instances may
# carry extra leading batch dims on every leaf (JAX rebuilds structs through
# the constructor when unflattening, so vmap/scan/tree-stack results all
# construct with batch-dim'd leaves). The schema machinery below compiles
# each data field's annotation into a spec of per-leaf shape constraints;
# batch inference and construction validation are then constraint solving
# over "which leading batch shape B is consistent with every leaf", and
# .shape, batched subscripting, and validation are all queries against the
# same solution. Specs form a small grammar:
#
#   spec ::= _ArraySpec      jaxtyping annotations and bare array classes;
#                            dims may be concrete, symbolic (rank-only), or
#                            unknown (a leading "..." variadic)
#          | _PyScalarSpec   python/numpy scalar classes: batch-agnostic
#          | _NoneSpec       None (as a union arm): batch-agnostic
#          | _ClassSpec      any other class: isinstance-checked, then
#                            recursed via the *value's* dynamic type (a
#                            nested struct/dataclass recurses its own
#                            schema; any other pytree constrains the batch
#                            as a prefix of every leaf's shape)
#          | _ContainerSpec  dict[k, v], list[t], tuple[t, ...], tuple[...]
#          | _UnionSpec      any union: the arm is decided by the value
#
# Plain scalar annotations (bool/int/float/complex) are sugar for the
# corresponding jaxtyping-over-ArrayLike union. Annotations that promise
# nothing about array leaves (Any, object, str, callables) are rejected:
# data fields hold array-leaved pytrees by definition (that is what being
# traced means); anything else belongs in static fields.


@dataclasses.dataclass(frozen=True)
class _ArraySpec:
    jaxtype: type       # the jaxtyping annotation as declared
    ndim: int | None    # element rank; None if unknown (leading variadic)
    min_ndim: int       # known trailing dims (== ndim when rank is known)
    fixed: tuple        # ((negative_offset, size), ...) concrete trailing dims
    names: tuple        # ((negative_offset, name), ...) symbolic trailing dims
    dtypes: tuple | None    # acceptable dtype names; None = any dtype

    def __str__(self):
        return getattr(self.jaxtype, "__name__", repr(self.jaxtype))


@dataclasses.dataclass(frozen=True)
class _PyScalarSpec:
    cls: type

    def __str__(self):
        return self.cls.__name__


@dataclasses.dataclass(frozen=True)
class _NoneSpec:
    def __str__(self):
        return "None"


@dataclasses.dataclass(frozen=True)
class _ClassSpec:
    cls: type

    def __str__(self):
        return self.cls.__name__


@dataclasses.dataclass(frozen=True)
class _ContainerSpec:
    kind: str       # "dict" | "list" | "tuple" | "tuple_variadic"
    elems: tuple    # child specs: one for dict/list/tuple_variadic, n for tuple

    def __str__(self):
        if self.kind == "dict":
            return f"dict[..., {self.elems[0]}]"
        elif self.kind == "list":
            return f"list[{self.elems[0]}]"
        elif self.kind == "tuple_variadic":
            return f"tuple[{self.elems[0]}, ...]"
        else:
            return f"tuple[{', '.join(str(e) for e in self.elems)}]"


@dataclasses.dataclass(frozen=True)
class _UnionSpec:
    arms: tuple

    def __str__(self):
        return " | ".join(str(arm) for arm in self.arms)


@dataclasses.dataclass(frozen=True)
class Schema:
    """
    The compiled form of a struct's data field annotations: a mapping from
    field names to shape-constraint specs. Build one with `strux.schema`.
    """
    cls: type
    fields: dict
    # precompiled validation plan for the common all-simple-fields case
    # (None when any field needs the general solver); see _fast_plan
    fast_plan: tuple | None = None

    def __str__(self):
        lines = [f"schema {self.cls.__name__}:"]
        for name, spec in self.fields.items():
            lines.append(f"  {name}: {spec}")
        return "\n".join(lines)


def schema(cls) -> Schema:
    """
    Compile a dataclass's data field annotations into a Schema (cached on
    the class). Works for strux structs and for other annotated dataclass
    pytrees (fields whose dataclasses metadata marks them static via
    `static=True` or `pytree_node=False` are excluded).

    Raises SchemaError if any data field's annotation cannot be compiled
    (e.g. Any, callables, unresolvable forward references).
    """
    if "_strux_schema" in cls.__dict__:
        return cls.__dict__["_strux_schema"]
    if not dataclasses.is_dataclass(cls):
        raise SchemaError(
            f"{cls.__name__} is not a dataclass; strux schemas describe "
            "annotated dataclass pytrees"
        )
    if "_data_fields" in vars(cls):
        data_fields = cls._data_fields
    else:
        data_fields = [
            f.name
            for f in dataclasses.fields(cls)
            if f.metadata.get("pytree_node", True)
            and not f.metadata.get("static", False)
        ]
    fields = {}
    for name in data_fields:
        hint, owner = _field_hint(cls, name)
        context = f"{cls.__name__}.{name}"
        hint = _resolve_hint(hint, owner=owner, cls=cls, context=context)
        fields[name] = _parse_hint(hint, context=context)
    result = Schema(cls=cls, fields=fields, fast_plan=_fast_plan(fields))
    cls._strux_schema = result
    return result


def _fast_plan(fields):
    """
    Precompile the validation plan for the common case: every field is a
    rank-determined array, or a scalar-or-rank-0-array union. Returns None
    (use the general solver) if any field is more complex. Plan entries:
    (name, dtype_names_frozenset_or_None, ndim_or_None, fixed) — ndim None
    marks a scalar-ish slot (python scalars skipped, arrays contribute
    their full shape as the batch).
    """
    plan = []
    for name, spec in fields.items():
        if isinstance(spec, _ArraySpec) and spec.ndim is not None:
            dtype_set = (
                _concrete_dtypes(spec.dtypes)
                if spec.dtypes is not None
                else None
            )
            plan.append((name, dtype_set, spec.ndim, spec.fixed))
        elif isinstance(spec, _UnionSpec) and all(
            isinstance(arm, _PyScalarSpec)
            or (
                isinstance(arm, _ArraySpec)
                and arm.ndim == 0
                and arm.dtypes is not None
            )
            for arm in spec.arms
        ):
            # scalar-ish: python scalar (batch-agnostic) or rank-0 array
            dtype_set = _concrete_dtypes(
                frozenset().union(
                    *(
                        arm.dtypes
                        for arm in spec.arms
                        if isinstance(arm, _ArraySpec)
                    )
                )
            )
            plan.append((name, dtype_set, None, ()))
        else:
            return None
    return tuple(plan)


def _concrete_dtypes(dtype_names):
    """
    Resolve dtype names to np.dtype objects for fast membership tests
    (dtype.name is computed dynamically by numpy and is far slower).
    Names that don't resolve to concrete dtypes (regex-style patterns,
    exotic kinds) are dropped: such values fail the fast path and are
    judged by the general solver instead.
    """
    dtypes = set()
    for dtype_name in dtype_names:
        try:
            dtypes.add(np.dtype(dtype_name))
        except TypeError:
            pass
    return frozenset(dtypes)


def _field_hint(cls, name):
    """
    Find a field's raw annotation and the class in whose body it was
    declared (annotations must be resolved in the *declaring* module's
    namespace, which under inheritance may differ from cls's own).
    """
    for klass in cls.__mro__:
        annotations = klass.__dict__.get("__annotations__", {})
        if name in annotations:
            return annotations[name], klass
    raise SchemaError(f"{cls.__name__}.{name}: no annotation found")


def _resolve_hint(hint, owner, cls, context):
    """
    Resolve a string annotation (e.g. under `from __future__ import
    annotations`) to a real type, in the declaring class's module namespace.
    Resolution happens lazily (at first schema use, not at decoration), so
    self-references and later definitions are already bound by the time
    they are needed.
    """
    if not isinstance(hint, str):
        return hint
    module = sys.modules.get(owner.__module__, None)
    globalns = getattr(module, "__dict__", {})
    localns = {owner.__name__: owner, cls.__name__: cls}
    try:
        return eval(hint, globalns, localns)
    except NameError as e:
        raise SchemaError(
            f"{context}: cannot resolve annotation {hint!r} ({e}); if the "
            "name is imported only under typing.TYPE_CHECKING, it must "
            "instead be available at runtime for strux to build the schema"
        ) from e


def _parse_hint(hint, context):
    """
    Compile one data field annotation into a spec. Raises SchemaError for
    annotations that don't promise an array-leaved pytree.
    """
    # unwrap Annotated[X, ...] to X
    if typing.get_origin(hint) is typing.Annotated:
        hint = typing.get_args(hint)[0]
    # reject annotations that promise nothing about array leaves
    if hint is typing.Any or hint is object:
        raise SchemaError(
            f"{context}: data fields must be array-leaved pytrees, but "
            f"{hint} promises nothing about its values; annotate the array "
            "structure, or mark the field static"
        )
    if hint in (str, bytes):
        raise SchemaError(
            f"{context}: {hint.__name__} is not pytree data; mark the "
            "field static"
        )
    origin = typing.get_origin(hint)
    if (
        hint is collections.abc.Callable
        or hint is typing.Callable
        or origin is collections.abc.Callable
    ):
        raise SchemaError(
            f"{context}: callables are not pytree data; mark the field "
            "static"
        )
    # None (as a union arm or a bare annotation)
    if hint is None or hint is type(None):
        return _NoneSpec()
    # unions (including Optional and jaxtyping-over-ArrayLike expansions);
    # scalar classes as explicit union arms are plain scalar arms (the
    # ArrayLike sugar below applies only to a bare scalar annotation)
    if origin is typing.Union or isinstance(hint, types.UnionType):
        return _UnionSpec(
            arms=tuple(
                _PyScalarSpec(cls=arm)
                if arm in (bool, int, float, complex)
                else _parse_hint(arm, context)
                for arm in typing.get_args(hint)
            ),
        )
    # jaxtyping array annotations
    if _is_jaxtype(hint):
        return _parse_jaxtype(hint, context)
    # plain scalar annotations are sugar for jaxtyping over ArrayLike
    if hint in (bool, int, float, complex):
        return _scalar_spec(hint)
    # containers of specs
    if origin is dict:
        _keytype, valtype = typing.get_args(hint)
        return _ContainerSpec(
            kind="dict",
            elems=(_parse_hint(valtype, context),),
        )
    if origin is list:
        (elemtype,) = typing.get_args(hint)
        return _ContainerSpec(
            kind="list",
            elems=(_parse_hint(elemtype, context),),
        )
    if origin is tuple:
        args = typing.get_args(hint)
        if len(args) == 2 and args[1] is Ellipsis:
            return _ContainerSpec(
                kind="tuple_variadic",
                elems=(_parse_hint(args[0], context),),
            )
        return _ContainerSpec(
            kind="tuple",
            elems=tuple(_parse_hint(arg, context) for arg in args),
        )
    # bare array classes: dtype and rank unknown
    if hint is jax.Array or hint is np.ndarray:
        return _parse_jaxtype(jaxtyping.Shaped[hint, "..."], context)
    # scalar classes appearing as union arms (e.g. numpy.number inside a
    # jaxtyping-over-ArrayLike expansion)
    if isinstance(hint, type) and issubclass(
        hint, (bool, int, float, complex, np.generic)
    ):
        return _PyScalarSpec(cls=hint)
    # any other class: an instance-checked pytree (nested struct, foreign
    # dataclass, abstract base of structs, or registered container); its
    # constraints are derived from the value's dynamic type
    if isinstance(hint, type):
        return _ClassSpec(cls=hint)
    raise SchemaError(
        f"{context}: unsupported data field annotation {hint!r}"
    )


def _parse_jaxtype(hint, context):
    if hint.index_variadic is not None and hint.index_variadic != 0:
        raise SchemaError(
            f"{context}: variadic dims are only supported leading (got "
            f"{hint.dim_str!r}); a field annotation describes one element, "
            "with any batch dims prepended in front"
        )
    tokens = hint.dim_str.split()
    if hint.index_variadic == 0:
        tokens = tokens[1:]
        ndim = None
    else:
        ndim = len(tokens)
    fixed = []
    names = []
    for i, token in enumerate(tokens):
        offset = i - len(tokens)    # negative index from the end
        if token.isdigit():
            fixed.append((offset, int(token)))
        elif token.isidentifier() and token != "_":
            names.append((offset, token))
        # other tokens (anonymous "_", broadcastable "#n", symbolic
        # expressions) contribute rank only
    dtypes = hint.dtypes if isinstance(hint.dtypes, tuple) else None
    return _ArraySpec(
        jaxtype=hint,
        ndim=ndim,
        min_ndim=len(tokens),
        fixed=tuple(fixed),
        names=tuple(names),
        dtypes=dtypes,
    )


_SCALAR_DTYPE_CLASSNAMES = {
    bool: "Bool",
    int: "Int",
    float: "Float",
    complex: "Complex",
}


@functools.lru_cache(maxsize=None)
def _scalar_spec(scalar_cls):
    # a plain scalar annotation is sugar for the corresponding
    # jaxtyping-over-ArrayLike union: the python scalar itself, or an
    # array (of any batch shape) of the matching dtype kind
    dtype_cls = getattr(jaxtyping, _SCALAR_DTYPE_CLASSNAMES[scalar_cls])
    hint = dtype_cls[jax.typing.ArrayLike, ""]
    context = f"scalar {scalar_cls.__name__}"
    arms = tuple(
        _parse_jaxtype(arm, context)
        if _is_jaxtype(arm)
        else _PyScalarSpec(cls=arm)
        for arm in typing.get_args(hint)
    )
    return _UnionSpec(arms=arms)


# # #
# Solving for the batch shape


# The candidate-set lattice: each (value, spec) pair yields the set of
# leading batch shapes B consistent with it — a finite frozenset of tuples,
# or _TOP (represented as None) meaning "unconstrained" (python scalars,
# None arms, empty containers). Validation asks whether the intersection
# across all fields is non-empty; .shape asks for its unique element.
_TOP = None


class _BatchMeet:
    """Accumulate per-field candidate sets, intersecting as we go."""

    def __init__(self):
        self.candidates = _TOP
        self.witnesses = []

    def add(self, path, candidates):
        if candidates is _TOP:
            return
        if self.candidates is _TOP:
            self.candidates = candidates
        else:
            met = self.candidates & candidates
            if not met:
                raise ValidationError(
                    "inconsistent batch shapes: "
                    f"{', '.join(self.witnesses)} admit batch shapes "
                    f"{_format_candidates(self.candidates)}, but {path} "
                    f"admits {_format_candidates(candidates)}"
                )
            self.candidates = met
        self.witnesses.append(path)


def _format_candidates(candidates):
    ordered = sorted(candidates, key=lambda b: (len(b), b))
    inner = ", ".join(str(b) for b in ordered[:4])
    if len(ordered) > 4:
        inner += ", ..."
    return "{" + inner + "}"


def _is_arraylike_value(value):
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _dtype_matches(dtype, patterns):
    if patterns is None:
        return True
    name = dtype.name
    if name in patterns:
        return True
    # zero-tangent dtype produced by autodiff for non-differentiable
    # (integer/bool) leaves; accept anywhere so gradients of structs with
    # such fields remain constructible
    if dtype == jax.dtypes.float0:
        return True
    for pattern in patterns:
        if pattern == "prng_key":
            if jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
                return True
        elif re.fullmatch(pattern, name):
            return True
    return False


def _candidates(value, spec, path):
    """
    The set of batch shapes consistent with this value under this spec
    (frozenset of tuples, or _TOP for "unconstrained"). Raises
    ValidationError if no batch shape is consistent.
    """
    if isinstance(spec, _ArraySpec):
        if not _is_arraylike_value(value):
            raise ValidationError(
                f"{path}: expected an array ({spec}), got "
                f"{type(value).__name__}"
            )
        if not _dtype_matches(value.dtype, spec.dtypes):
            raise ValidationError(
                f"{path}: expected dtype kind {spec} but got dtype "
                f"{value.dtype.name}"
            )
        shape = tuple(value.shape)
        if len(shape) < spec.min_ndim:
            raise ValidationError(
                f"{path}: expected at least {spec.min_ndim} element dims "
                f"({spec}), got shape {shape}"
            )
        for offset, size in spec.fixed:
            if shape[len(shape) + offset] != size:
                raise ValidationError(
                    f"{path}: expected element dims matching {spec}, got "
                    f"shape {shape} (trailing dim {offset} should be "
                    f"{size})"
                )
        if spec.ndim is None:
            return frozenset(
                shape[:k] for k in range(len(shape) - spec.min_ndim + 1)
            )
        return frozenset((shape[:len(shape) - spec.ndim],))
    elif isinstance(spec, _PyScalarSpec):
        if isinstance(value, spec.cls):
            return _TOP     # a scalar broadcasts: batch-agnostic
        raise ValidationError(
            f"{path}: expected {spec.cls.__name__}, got "
            f"{type(value).__name__}"
        )
    elif isinstance(spec, _NoneSpec):
        if value is None:
            return _TOP
        raise ValidationError(
            f"{path}: expected None, got {type(value).__name__}"
        )
    elif isinstance(spec, _ClassSpec):
        if not isinstance(value, spec.cls):
            raise ValidationError(
                f"{path}: expected an instance of {spec.cls.__name__}, got "
                f"{type(value).__name__}"
            )
        value_cls = type(value)
        if dataclasses.is_dataclass(value_cls):
            # an annotated dataclass (nested struct or foreign): recurse
            # via the *value's* schema (which may be a subclass of the
            # annotated class, with more fields)
            meet = _BatchMeet()
            for name, subspec in schema(value_cls).fields.items():
                meet.add(
                    f"{path}.{name}",
                    _candidates(getattr(value, name), subspec, f"{path}.{name}"),
                )
            return meet.candidates
        # any other pytree: every array leaf constrains the batch as a
        # prefix of its shape; scalar leaves are batch-agnostic
        meet = _BatchMeet()
        for keypath, leaf in jax.tree.flatten_with_path(value)[0]:
            leafpath = path + "".join(str(k) for k in keypath)
            if _is_arraylike_value(leaf):
                shape = tuple(leaf.shape)
                meet.add(
                    leafpath,
                    frozenset(shape[:k] for k in range(len(shape) + 1)),
                )
            elif isinstance(leaf, (bool, int, float, complex)):
                continue
            else:
                raise ValidationError(
                    f"{leafpath}: value of type {type(leaf).__name__} "
                    "inside a data field is not an array; restructure, or "
                    "mark the field static"
                )
        return meet.candidates
    elif isinstance(spec, _ContainerSpec):
        expected_cls = dict if spec.kind == "dict" else (
            list if spec.kind == "list" else tuple
        )
        if not isinstance(value, expected_cls):
            raise ValidationError(
                f"{path}: expected {spec}, got {type(value).__name__}"
            )
        if spec.kind == "dict":
            children = [(f"{path}[{k!r}]", v, spec.elems[0]) for k, v in value.items()]
        elif spec.kind in ("list", "tuple_variadic"):
            children = [
                (f"{path}[{i}]", v, spec.elems[0]) for i, v in enumerate(value)
            ]
        else:
            if len(value) != len(spec.elems):
                raise ValidationError(
                    f"{path}: expected {spec} (length {len(spec.elems)}), "
                    f"got length {len(value)}"
                )
            children = [
                (f"{path}[{i}]", v, e)
                for i, (v, e) in enumerate(zip(value, spec.elems))
            ]
        meet = _BatchMeet()
        for childpath, child, childspec in children:
            meet.add(childpath, _candidates(child, childspec, childpath))
        return meet.candidates
    elif isinstance(spec, _UnionSpec):
        arm_candidates = []
        arm_failures = []
        for arm in spec.arms:
            try:
                arm_candidates.append(_candidates(value, arm, path))
            except ValidationError as e:
                arm_failures.append(f"as {arm}: {e}")
        if not arm_candidates:
            raise ValidationError(
                f"{path}: value matches no arm of {spec}: "
                + "; ".join(arm_failures)
            )
        if any(c is _TOP for c in arm_candidates):
            return _TOP
        return frozenset().union(*arm_candidates)
    else:
        raise AssertionError(f"unknown spec {spec!r}")


def _validate_struct(instance):
    """
    Check a freshly-constructed struct against its schema: array kinds and
    element dims as annotated, and a consistent leading batch shape across
    all data fields. Raises ValidationError (or SchemaError if the
    annotations themselves cannot be compiled).
    """
    cls = type(instance)
    sch = schema(cls)
    # fast path: simple fields validated with plain tuple arithmetic; any
    # anomaly (including acceptable oddities like float0 cotangents and
    # prng-key dtypes) falls through to the general solver, which either
    # accepts them or raises with a precise message
    plan = sch.fast_plan
    if plan is not None and _fast_validate(instance, plan):
        return
    meet = _BatchMeet()
    for name, spec in sch.fields.items():
        path = f"{cls.__name__}.{name}"
        meet.add(path, _candidates(getattr(instance, name), spec, path))


def _fast_validate(instance, plan):
    batch = None
    for name, dtype_set, ndim, fixed in plan:
        value = getattr(instance, name)
        if ndim is None and isinstance(value, (bool, int, float, complex)):
            continue    # scalar-ish slot with a python scalar
        try:
            shape = value.shape
            dtype = value.dtype
        except AttributeError:
            return False
        if dtype_set is not None and dtype not in dtype_set:
            return False
        if ndim is None:
            value_batch = tuple(shape)
        else:
            cut = len(shape) - ndim
            if cut < 0:
                return False
            for offset, size in fixed:
                if shape[len(shape) + offset] != size:
                    return False
            value_batch = tuple(shape[:cut])
        if batch is None:
            batch = value_batch
        elif value_batch != batch:
            return False
    return True


# # # 
# Pretty printing


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


# # # 
# Batch annotations


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

    Each field's expansion is derived from its schema spec. Fields whose
    element rank is unknown (leading-variadic annotations, bare array
    classes, abstract pytree classes) are checked for *consistency with*
    the batch dims (the leading dims must be there; where the element ends
    cannot be known), rather than certainty.
    """
    field_hints = {}
    for name, spec in schema(struct_cls).fields.items():
        context = f'{struct_cls.__name__}["{dims}"].{name}'
        field_hints[name] = _expand_spec(spec, dims, context)
    return _StructAnnotationMeta(
        f'{struct_cls.__name__}["{dims}"]',
        (),
        {
            '_struct_type': struct_cls,
            '_dims': dims,
            '_field_hints': field_hints,
        },
    )


def _expand_spec(spec, dims, context):
    """
    Turn a field spec into an isinstance-checkable object (a class, or a
    flat tuple of classes for unions) representing the field with batch
    dims prepended. Returns None for specs that cannot carry batch dims
    (python scalar arms of a union when dims is non-empty).
    """
    if isinstance(spec, _ArraySpec):
        jt = spec.jaxtype
        return jt.dtype[jt.array_type, f"{dims} {jt.dim_str}".strip()]
    elif isinstance(spec, _PyScalarSpec):
        if dims.strip():
            return None     # a python scalar cannot carry batch dims
        return spec.cls
    elif isinstance(spec, _NoneSpec):
        return type(None)   # a batch of Nones is still (structurally) None
    elif isinstance(spec, _ClassSpec):
        if dataclasses.is_dataclass(spec.cls):
            return _make_struct_annotation(spec.cls, dims)
        # abstract base classes and other registered pytrees: check the
        # class and require every array leaf to carry the batch dims
        # (python scalar leaves are batch-agnostic, as in the solver)
        leaf_hints = (
            jaxtyping.Shaped[jax.Array, f"{dims} ..."],
            jaxtyping.Shaped[np.ndarray, f"{dims} ..."],
            bool, int, float, complex,
        )
        def check_node(value, _cls=spec.cls, _leaf_hints=leaf_hints):
            if not isinstance(value, _cls):
                return False
            value_cls = type(value)
            if dataclasses.is_dataclass(value_cls):
                return isinstance(
                    value, _make_struct_annotation(value_cls, dims),
                )
            return all(
                isinstance(leaf, _leaf_hints)
                for leaf in jax.tree.leaves(value)
            )
        return _make_checker(f'{spec.cls.__name__}["{dims}"]', check_node)
    elif isinstance(spec, _ContainerSpec):
        elem_hints = tuple(
            _expand_spec(elem, dims, context) for elem in spec.elems
        )
        if any(hint is None for hint in elem_hints):
            raise SchemaError(
                f"{context}: container elements cannot carry batch dims"
            )
        def check_container(value, _spec=spec, _elem_hints=elem_hints):
            if _spec.kind == "dict":
                return isinstance(value, dict) and all(
                    isinstance(v, _elem_hints[0]) for v in value.values()
                )
            elif _spec.kind == "list":
                return isinstance(value, list) and all(
                    isinstance(v, _elem_hints[0]) for v in value
                )
            elif _spec.kind == "tuple_variadic":
                return isinstance(value, tuple) and all(
                    isinstance(v, _elem_hints[0]) for v in value
                )
            else:
                return (
                    isinstance(value, tuple)
                    and len(value) == len(_elem_hints)
                    and all(
                        isinstance(v, hint)
                        for v, hint in zip(value, _elem_hints)
                    )
                )
        return _make_checker(f'{spec}["{dims}"]', check_container)
    elif isinstance(spec, _UnionSpec):
        arms = []
        for arm in spec.arms:
            expanded = _expand_spec(arm, dims, context)
            if expanded is None:
                continue
            if isinstance(expanded, tuple):
                arms.extend(expanded)
            else:
                arms.append(expanded)
        if not arms:
            raise SchemaError(
                f"{context}: no arm of {spec} can carry batch dims"
            )
        if len(arms) == 1:
            return arms[0]
        return tuple(arms)
    else:
        raise AssertionError(f"unknown spec {spec!r}")


class _CheckerMeta(type):
    """Metaclass giving a synthetic annotation a custom isinstance check."""
    def __instancecheck__(cls, instance):
        return cls._check(instance)


def _make_checker(name, check_fn):
    return _CheckerMeta(name, (), {"_check": staticmethod(check_fn)})


class _StructAnnotationMeta(type):
    """Metaclass for batched struct type annotations with isinstance support."""
    def __instancecheck__(cls, instance):
        if not isinstance(instance, cls._struct_type):
            return False
        instance_cls = type(instance)
        if instance_cls is not cls._struct_type and dataclasses.is_dataclass(
            instance_cls
        ):
            # a subclass instance: check against the subclass's own schema
            # (it may declare more fields than the annotated base)
            return isinstance(
                instance, _make_struct_annotation(instance_cls, cls._dims),
            )
        for field_name, expected_type in cls._field_hints.items():
            if not isinstance(getattr(instance, field_name), expected_type):
                return False
        return True


# # # 
# Shapes and indexing batched structs


def tree_shape(tree) -> tuple[int, ...]:
    """
    Return the batch shape of a struct: the leading dimensions beyond each
    field's element annotation.

    The batch shape is solved from the schema (see `strux.schema`): each
    field constrains the possible batch shapes, and all data fields must
    agree. Fields whose element rank is unknown (leading-variadic
    annotations, bare array classes, abstract pytree classes) and
    batch-agnostic values (python scalars, None) are consistent with many
    batch shapes; if the fields that *do* determine ranks don't pin the
    batch down to a single shape, this raises ValueError listing the
    consistent candidates — annotate element dims on at least one field to
    resolve it. A struct with only batch-agnostic values has batch shape ().
    """
    cls = type(tree)
    meet = _BatchMeet()
    for name, spec in schema(cls).fields.items():
        path = f"{cls.__name__}.{name}"
        meet.add(path, _candidates(getattr(tree, name), spec, path))
    if meet.candidates is _TOP:
        return ()
    if len(meet.candidates) == 1:
        return next(iter(meet.candidates))
    raise ValueError(
        f"{cls.__name__}: batch shape under-determined: consistent "
        f"candidates {_format_candidates(meet.candidates)}; annotate "
        "element dims on at least one field to determine the batch shape"
    )


def tree_dims(tree) -> dict:
    """
    Bind the symbolic dimension names appearing in a struct's field
    annotations to their sizes in this instance, and check consistency.

    Names share one namespace across the whole struct, including nested
    structs and containers: annotate `weights: Float[Array, "n_in n_out"]`
    and `biases: Float[Array, "n_out"]` and `tree_dims(net)` returns
    `{"n_in": ..., "n_out": ...}` — or raises ValidationError if the two
    `n_out`s disagree. (Construction does not enforce this; symbolic dims
    are rank-only at construction. This query is the prototype of
    eventual cross-field named-dim checking.)
    """
    # the batch shape is needed to pick union arms; without a determined
    # batch, ambiguous unions are skipped rather than guessed
    try:
        batch = tree_shape(tree)
    except ValueError:
        batch = None
    bindings = {}
    cls = type(tree)
    for name, spec in schema(cls).fields.items():
        _bind_dims(
            getattr(tree, name), spec, batch, bindings, f"{cls.__name__}.{name}",
        )
    return {name: size for name, (size, _) in bindings.items()}


def _bind_dims(value, spec, batch, bindings, path):
    if isinstance(spec, _ArraySpec):
        if not spec.names or not _is_arraylike_value(value):
            return
        shape = tuple(value.shape)
        for offset, dim_name in spec.names:
            size = shape[len(shape) + offset]
            if dim_name in bindings:
                previous_size, previous_path = bindings[dim_name]
                if previous_size != size:
                    raise ValidationError(
                        f"inconsistent dim '{dim_name}': {previous_size} at "
                        f"{previous_path}, but {size} at {path}"
                    )
            else:
                bindings[dim_name] = (size, path)
    elif isinstance(spec, _ClassSpec):
        value_cls = type(value)
        if dataclasses.is_dataclass(value_cls):
            for name, subspec in schema(value_cls).fields.items():
                _bind_dims(
                    getattr(value, name), subspec, batch, bindings,
                    f"{path}.{name}",
                )
    elif isinstance(spec, _ContainerSpec):
        if isinstance(value, dict):
            for k, v in value.items():
                _bind_dims(v, spec.elems[0], batch, bindings, f"{path}[{k!r}]")
        elif isinstance(value, (list, tuple)):
            if spec.kind == "tuple" and len(value) == len(spec.elems):
                elem_specs = spec.elems
            else:
                elem_specs = (spec.elems[0],) * len(value)
            for i, (v, e) in enumerate(zip(value, elem_specs)):
                _bind_dims(v, e, batch, bindings, f"{path}[{i}]")
    elif isinstance(spec, _UnionSpec):
        # bind through the arm this value inhabits, when determinable
        matching = []
        for arm in spec.arms:
            try:
                candidates = _candidates(value, arm, path)
            except ValidationError:
                continue
            if candidates is _TOP or batch is None or batch in candidates:
                matching.append(arm)
        if len(matching) == 1:
            _bind_dims(value, matching[0], batch, bindings, path)
    # scalar and None specs carry no named dims


def tree_getitem(tree, index):
    """Index into the batch dimensions of a struct."""
    return jax.tree.map(lambda x: x[index], tree)


def tree_size(tree) -> int:
    """Calculates the total number of parameters in the PyTree."""
    return sum(jnp.size(x) for x in jax.tree.leaves(tree))


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
    absent restores as None). It cannot reconstruct polymorphic fields
    (annotated with a base class but holding subclass instances) — restore
    those with an instance template.

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
        # keys strictly below this path: the first structured arm that fits
        if any(key.startswith(path + "/") for key in d):
            failures = []
            for arm in spec.arms:
                if isinstance(arm, (_ClassSpec, _ContainerSpec)):
                    try:
                        return _build_value(arm, d, statics, path)
                    except (KeyError, TypeError) as e:
                        failures.append(f"as {arm}: {e}")
            raise KeyError(
                f"cannot reconstruct {path!r} under any arm of {spec}: "
                + "; ".join(failures)
            )
        # no data at all: the None arm, if there is one
        if any(isinstance(arm, _NoneSpec) for arm in spec.arms):
            return None, set()
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
# Serialisation


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
