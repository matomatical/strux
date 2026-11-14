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
    """An annotation cannot describe supported numerical pytree data.

    Compilation is lazy so forward references can resolve after class
    definition. Annotation construction, checking and shape queries may
    require compilation; assigning values does not.
    """


class ValidationError(TypeError):
    """A value fails an explicit description or batch-shape constraint."""


# Specifications preserve annotation structure independently of value
# validation. Array ranks support batch inference; the original jaxtyping
# annotation remains available to the checker and Map callbacks. Class
# references are expanded lazily, including recursive dataclasses.


@dataclasses.dataclass(frozen=True)
class _ArraySpec:
    jaxtype: type  # the jaxtyping annotation as declared
    ndim: int | None  # element rank; None if unknown (leading variadic)
    min_ndim: int  # known trailing dims (== ndim when rank is known)
    fixed: tuple  # ((negative_offset, size), ...) concrete trailing dims
    names: tuple  # ((negative_offset, name), ...) symbolic trailing dims
    dtypes: tuple | None  # acceptable dtype names; None = any dtype

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
    dims: tuple = ()  # element-level batch dimensions
    transforms: tuple = ()

    def __str__(self):
        if self.dims:
            return f'Struct[{self.cls.__name__}, "{" ".join(self.dims)}"]'
        return self.cls.__name__


@dataclasses.dataclass(frozen=True)
class _ContainerSpec:
    kind: str  # "dict" | "list" | "tuple" | "tuple_variadic"
    elems: tuple  # child specs: one for dict/list/tuple_variadic, n for tuple
    keytype: object = object

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
    # Whether the default fields mention local symbolic dimensions.
    has_names: bool = False

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
    result = Schema(
        cls=cls,
        fields=fields,
        has_names=any(_spec_has_names(spec) for spec in fields.values()),
    )
    cls._strux_schema = result
    return result


def _spec_has_names(spec):
    """Does this spec (not counting nested dataclasses, whose names are
    their own class's scope) mention any symbolic dim names?"""
    if isinstance(spec, _ArraySpec):
        return bool(spec.names)
    if isinstance(spec, (_ContainerSpec, _UnionSpec)):
        children = spec.elems if isinstance(spec, _ContainerSpec) else spec.arms
        return any(_spec_has_names(child) for child in children)
    return False


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
    from strux.annotate import Batch, _result_spec

    if isinstance(hint, Batch):
        return _result_spec(hint)
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
            f"{context}: {hint.__name__} is not pytree data; mark the " "field static"
        )
    origin = typing.get_origin(hint)
    if (
        hint is collections.abc.Callable
        or hint is typing.Callable
        or origin is collections.abc.Callable
    ):
        raise SchemaError(
            f"{context}: callables are not pytree data; mark the field " "static"
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
                (
                    _PyScalarSpec(cls=arm)
                    if arm in (bool, int, float, complex)
                    else _parse_hint(arm, context)
                )
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
            keytype=_keytype,
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
    # Array data needs explicit dtype/shape requirements.
    if hint is jax.Array or hint is np.ndarray:
        raise SchemaError(
            f"{context}: bare array classes need a jaxtyping dtype and shape "
            'annotation, such as Float[Array, "features"]'
        )
    # scalar classes appearing as union arms (e.g. numpy.number inside a
    # jaxtyping-over-ArrayLike expansion)
    if isinstance(hint, type) and issubclass(
        hint, (bool, int, float, complex, np.generic)
    ):
        return _PyScalarSpec(cls=hint)
    form = _struct_form(hint)
    if form is not None:
        from strux.annotate import _Operation

        target, transformations = form
        if all(isinstance(op, str) for op in transformations):
            return _ClassSpec(cls=target, dims=_form_dims(transformations))
        return _ClassSpec(
            cls=target,
            transforms=tuple(_Operation(op) for op in transformations),
        )
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
    raise SchemaError(f"{context}: unsupported data field annotation {hint!r}")


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
        offset = i - len(tokens)  # negative index from the end
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
        _parse_jaxtype(arm, context) if _is_jaxtype(arm) else _PyScalarSpec(cls=arm)
        for arm in typing.get_args(hint)
    )
    return _UnionSpec(arms=arms)


def _struct_form(hint):
    """
    The `(target, functors)` a `Struct[...]` annotation stands for, or
    None if `hint` is not one.
    """
    if not isinstance(hint, type):
        return None
    target = getattr(hint, "_struct_type", None)
    functors = getattr(hint, "_functors", None)
    if target is None or functors is None:
        return None
    return target, functors


def _form_dims(functors):
    """
    The dims a pipeline of dims-string functors declares, as a flat tuple
    of jaxtyping dim tokens. Functors apply left to right and each
    prepends its dims in front of the previous, so the tokens read
    right-to-left across the strings.
    """
    return tuple(token for dim_str in reversed(functors) for token in dim_str.split())
