"""
Solving for the batch shape: given a struct instance and its schema, which
leading batch shapes are consistent with every data field? Construction
validation asks whether any shape is; the shape queries ask for the unique
one.
"""

import dataclasses
import re

import jax

from strux.schema import (
    schema,
    ValidationError,
    _ArraySpec,
    _PyScalarSpec,
    _NoneSpec,
    _ClassSpec,
    _ContainerSpec,
    _UnionSpec,
)


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
