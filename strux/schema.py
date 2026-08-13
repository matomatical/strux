"""
Compiling a struct's field annotations into a schema: a mapping from data
field names to shape-constraint specs, plus the errors raised when
compilation or validation fails.
"""

import collections.abc
import dataclasses
import functools
import inspect
import sys
import types
import typing

import jax
import jaxtyping
import numpy as np


def _is_jaxtype(hint) -> bool:
    """
    Is this type hint a jaxtyping array annotation? Jaxtyping is the only
    supported array annotation framework.
    """
    return isinstance(hint, type) and issubclass(hint, jaxtyping.AbstractArray)


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
    # inspect.get_annotations rather than a raw __dict__ lookup: it likewise
    # returns only the class's *own* annotations (never inherited ones), but
    # also handles deferred annotations (PEP 649, python 3.14+), where the
    # class dict carries an __annotate__ function instead of a ready dict
    for klass in cls.__mro__:
        try:
            annotations = inspect.get_annotations(klass)
        except NameError as e:
            raise SchemaError(
                f"{cls.__name__}.{name}: cannot evaluate deferred "
                f"annotations of {klass.__name__} ({e}); if the name is "
                "imported only under typing.TYPE_CHECKING, it must instead "
                "be available at runtime for strux to build the schema"
            ) from e
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
    # the declaring class's type parameters (class C[T]: ...) live in a
    # lexical scope of their own, not the module globals; bind them so
    # annotations like tuple[C[T], ...] resolve
    localns.update(
        {param.__name__: param for param in getattr(owner, "__type_params__", ())}
    )
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
    # generic aliases over ordinary classes (e.g. RewardFn[EnvT]): type
    # parameters are erased at runtime, so the constraint is the origin
    # class, recursed like any class annotation
    if isinstance(origin, type):
        return _parse_hint(origin, context)
    # type variables: a bounded one promises its bound; an unbounded one
    # promises nothing about its values
    if isinstance(hint, typing.TypeVar):
        if hint.__bound__ is None:
            raise SchemaError(
                f"{context}: unbounded type variable {hint!r} promises "
                "nothing about its values; give it a bound "
                "(class C[T: Bound]) or mark the field static"
            )
        return _parse_hint(hint.__bound__, context)
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
