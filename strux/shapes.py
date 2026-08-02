"""
Shapes and indexing for (batched) structs: the batch shape, symbolic dim
binding, batch indexing, and total leaf count.
"""

import dataclasses

import jax
import jax.numpy as jnp

from strux.schema import (
    schema,
    ValidationError,
    _ArraySpec,
    _ClassSpec,
    _ContainerSpec,
    _UnionSpec,
)
from strux.batch import (
    _TOP,
    _BatchMeet,
    _candidates,
    _format_candidates,
    _is_arraylike_value,
)


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
