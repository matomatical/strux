"""
Batch annotations: `Cls["batch"]` synthetic types whose isinstance checks
verify a struct as if every field annotation had the batch dims prepended.
"""

import dataclasses
import functools

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
)


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
