"""
The Struct type form: `Struct[Cls, ...]` synthetic types whose isinstance
checks verify a struct as the image of type-level transformations of its
field annotations — batch dims prepended, dtypes changed, leaves collapsed.

To static checkers (under typing.TYPE_CHECKING) `strux.Struct` is
`typing.Annotated`, so the same expression is legal in annotation position:
checkers see the underlying struct class and ignore the transformation
arguments, which are metadata to them and instructions to this module.
"""

import dataclasses
import functools
import typing

import jax
import jaxtyping
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
    _scalar_spec,
)


# # #
# Functors
#
# A functor is the type-level shadow of a leaf-wise value transformation:
# it says what happens to each field annotation. Two are built in, plus
# dims strings:
#
#   "b t"          prepend these batch dims to every leaf (the image of
#                  stacking/vmapping); "" asserts the element exactly
#                  (rank-exact) and "..." allows any batch rank
#   astype(kind)   every leaf keeps its dims but takes the scalar kind
#                  (the image of e.g. tree.map(operator.eq, x, y))
#   mapped(kind)   every leaf becomes a scalar of the kind (the image of
#                  a full reduction, e.g. tree.map(jnp.array_equal, x, y)
#                  or tree.map(jnp.count_nonzero, x))
#
# Struct[Cls, f1, f2, ...] applies the functors left to right (application
# order). A constant map (mapped) absorbs everything before it, exactly as
# the reduction it shadows would.


_KIND_DTYPES = {
    bool: jaxtyping.Bool,
    int: jaxtyping.Int,
    float: jaxtyping.Float,
    complex: jaxtyping.Complex,
}


def _require_scalar_kind(kind, functor_name):
    if kind not in _KIND_DTYPES:
        raise TypeError(
            f"{functor_name} takes a python scalar class "
            f"(bool, int, float, or complex), got {kind!r}"
        )


@dataclasses.dataclass(frozen=True)
class astype:
    """
    Functor: every array leaf keeps its dims but takes on the scalar kind
    (python scalar leaves become that python scalar). The image of a
    leaf-wise dtype-changing map, e.g. `tree.map(operator.eq, x, y)` has
    type `Struct[Cls, astype(bool)]`.
    """

    kind: type

    def __post_init__(self):
        _require_scalar_kind(self.kind, "astype")

    def __repr__(self):
        return f"astype({self.kind.__name__})"


@dataclasses.dataclass(frozen=True)
class mapped:
    """
    Functor: every leaf becomes a scalar of the kind (a python scalar or a
    rank-0 array, the same sugar as a plain scalar field annotation). The
    image of a leaf-wise full reduction, e.g.
    `tree.map(jnp.array_equal, x, y)` has type `Struct[Cls, mapped(bool)]`
    and `tree.map(jnp.count_nonzero, x)` has type
    `Struct[Cls, mapped(int)]`.
    """

    kind: type

    def __post_init__(self):
        _require_scalar_kind(self.kind, "mapped")

    def __repr__(self):
        return f"mapped({self.kind.__name__})"


def _functors_str(functors):
    return ", ".join(repr(functor) for functor in functors)


# # #
# The Struct form


class Struct:
    """
    The type form for structs: `Struct[Cls, functors...]` produces a
    synthetic type whose isinstance checks verify an instance of `Cls` as
    the image of the functors (dims strings prepend batch dims; see the
    functor classes), applied left to right.

    For example, given:

        @strux.struct
        class Env:
            pos: Int[Array, "2"]
            walls: Bool[Array, "h w"]

    then `Struct[Env, "batch"]` checks as if against:

        @strux.struct
        class Envs:
            pos: Int[Array, "batch 2"]
            walls: Bool[Array, "batch h w"]

    To static type checkers `strux.Struct` is `typing.Annotated`, so these
    forms are legal in annotation position, where checkers read them as
    plain `Env` (batch dims are beyond static checking, like jaxtyping's
    shape strings) while runtime checkers (isinstance, beartype under
    `@jaxtyped`) enforce the full form.

    Fields whose element rank is unknown (leading-variadic annotations,
    bare array classes, abstract pytree classes) are checked for
    *consistency with* the functor image (where the element ends cannot be
    known), rather than certainty.
    """

    def __init__(self):
        raise TypeError(
            "Struct is a type form: subscript it (Struct[Cls, ...]) "
            "rather than instantiating it"
        )

    def __class_getitem__(cls, item):
        if not isinstance(item, tuple) or len(item) < 2:
            raise TypeError(
                "Struct[...] takes a struct class followed by at least one "
                "transformation (a dims string or functor); for the element "
                'type alone use the class itself, or Struct[Cls, ""] for a '
                "rank-exact check"
            )
        target, *functors = item
        # a generic alias parameterises the class for static checkers; at
        # runtime the parameters are erased and the origin is the check
        origin = typing.get_origin(target)
        if isinstance(origin, type):
            target = origin
        if not (isinstance(target, type) and dataclasses.is_dataclass(target)):
            raise TypeError(
                f"Struct[...] expects a struct (or other annotated "
                f"dataclass pytree) class first, got {target!r}"
            )
        for functor in functors:
            if not isinstance(functor, (str, astype, mapped)):
                raise TypeError(
                    f"Struct[...] transformations must be dims strings or "
                    f"functors (strux.astype, strux.mapped), got {functor!r}"
                )
        return _make_struct_annotation(target, tuple(functors))


@functools.lru_cache(maxsize=None)
def _make_struct_annotation(struct_cls, functors):
    """
    Build the synthetic type for `Struct[struct_cls, *functors]`: each
    field's schema spec is lowered to an isinstance-checkable hint for its
    image under the functor pipeline.
    """
    field_hints = {}
    for name, spec in schema(struct_cls).fields.items():
        context = f"Struct[{struct_cls.__name__}, {_functors_str(functors)}].{name}"
        field_hints[name] = _expand_spec(spec, functors, context)
    return _StructAnnotationMeta(
        f"Struct[{struct_cls.__name__}, {_functors_str(functors)}]",
        (),
        {
            '_struct_type': struct_cls,
            '_functors': functors,
            '_field_hints': field_hints,
        },
    )


def _expand_spec(spec, functors, context):
    """
    Lower a field spec plus a functor pipeline to an isinstance-checkable
    object (a class, or a flat tuple of classes for unions) representing
    the field's image. Returns None for images that cannot exist (python
    scalar arms of a union under nonempty prepended dims).
    """
    # a constant map replaces whatever leaf spec came before it (structure
    # — containers and nested structs — is preserved by tree.map and
    # handled below); cut at the last one
    if not isinstance(spec, (_ClassSpec, _ContainerSpec, _NoneSpec)):
        for i in reversed(range(len(functors))):
            if isinstance(functors[i], mapped):
                spec = _scalar_spec(functors[i].kind)
                functors = functors[i + 1:]
                break
    if isinstance(spec, _ArraySpec):
        jt = spec.jaxtype
        dtype_cls = jt.dtype
        dims = jt.dim_str
        for functor in functors:
            if isinstance(functor, str):
                dims = f"{functor} {dims}".strip()
            else:  # astype
                dtype_cls = _KIND_DTYPES[functor.kind]
        return dtype_cls[jt.array_type, dims]
    elif isinstance(spec, _PyScalarSpec):
        cls = spec.cls
        for functor in functors:
            if isinstance(functor, str):
                if functor.strip():
                    return None  # a python scalar cannot carry batch dims
            else:  # astype
                cls = functor.kind
        return cls
    elif isinstance(spec, _NoneSpec):
        # None is an empty subtree: leaf-wise maps never touch it, and a
        # batch of Nones is still (structurally) None
        return type(None)
    elif isinstance(spec, _ClassSpec):
        if dataclasses.is_dataclass(spec.cls):
            return _make_struct_annotation(spec.cls, functors)
        # abstract base classes and other registered pytrees: check the
        # class and require every array leaf to satisfy the functor image
        # (python scalar leaves are batch-agnostic, as in the solver)
        leaf_hints = _leaf_hints(functors)
        def check_node(value, _cls=spec.cls, _leaf_hints=leaf_hints,
                       _functors=functors):
            if not isinstance(value, _cls):
                return False
            value_cls = type(value)
            if dataclasses.is_dataclass(value_cls):
                return isinstance(
                    value, _make_struct_annotation(value_cls, _functors),
                )
            return all(
                isinstance(leaf, _leaf_hints)
                for leaf in jax.tree.leaves(value)
            )
        return _make_checker(
            f"{spec.cls.__name__}[{_functors_str(functors)}]", check_node,
        )
    elif isinstance(spec, _ContainerSpec):
        elem_hints = tuple(
            _expand_spec(elem, functors, context) for elem in spec.elems
        )
        if any(hint is None for hint in elem_hints):
            raise SchemaError(
                f"{context}: container elements cannot carry this image"
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
        return _make_checker(
            f"{spec}[{_functors_str(functors)}]", check_container,
        )
    elif isinstance(spec, _UnionSpec):
        arms = []
        for arm in spec.arms:
            expanded = _expand_spec(arm, functors, context)
            if expanded is None:
                continue
            if isinstance(expanded, tuple):
                arms.extend(expanded)
            else:
                arms.append(expanded)
        # dedupe (functor images of distinct arms may coincide)
        arms = list(dict.fromkeys(arms))
        if not arms:
            raise SchemaError(
                f"{context}: no arm of {spec} can carry this image"
            )
        if len(arms) == 1:
            return arms[0]
        return tuple(arms)
    else:
        raise AssertionError(f"unknown spec {spec!r}")


def _leaf_hints(functors):
    """
    The functor image of an unknown leaf (for pytree fields whose element
    structure the schema cannot see): any array carrying the image dims
    and dtype, or (batch-agnostic) a python scalar of the image kind.
    """
    dtype_cls = jaxtyping.Shaped
    dims = "..."
    scalar_arms = (bool, int, float, complex)
    for functor in functors:
        if isinstance(functor, str):
            dims = f"{functor} {dims}".strip()
        elif isinstance(functor, astype):
            dtype_cls = _KIND_DTYPES[functor.kind]
            scalar_arms = (functor.kind,)
        else:  # mapped
            dtype_cls = _KIND_DTYPES[functor.kind]
            dims = ""
            scalar_arms = (functor.kind,)
    return (
        dtype_cls[jax.Array, dims],
        dtype_cls[np.ndarray, dims],
        *scalar_arms,
    )


class _CheckerMeta(type):
    """Metaclass giving a synthetic annotation a custom isinstance check."""
    def __instancecheck__(cls, instance):
        return cls._check(instance)


def _make_checker(name, check_fn):
    return _CheckerMeta(name, (), {"_check": staticmethod(check_fn)})


class _StructAnnotationMeta(type):
    """Metaclass for Struct-form annotations with isinstance support."""
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
                instance,
                _make_struct_annotation(instance_cls, cls._functors),
            )
        for field_name, expected_type in cls._field_hints.items():
            if not isinstance(getattr(instance, field_name), expected_type):
                return False
        return True
