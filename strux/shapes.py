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
    _candidates,
    _format_candidates,
    _is_arraylike_value,
    _solved_candidates,
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

    The solved candidates are cached on the instance (structs are
    immutable), so repeated queries cost one attribute lookup.
    """
    candidates = _solved_candidates(tree)
    if candidates is _TOP:
        return ()
    if len(candidates) == 1:
        return next(iter(candidates))
    raise ValueError(
        f"{type(tree).__name__}: batch shape under-determined: consistent "
        f"candidates {_format_candidates(candidates)}; annotate "
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
    """
    Index into the batch dimensions of a struct.

    An unbatched struct (batch shape ()) refuses indexing with TypeError,
    and integer indices are bounds-checked against the batch shape
    (IndexError out of bounds, negative indices in the python style) — so
    indexing can never silently reach into *element* dimensions. Slices
    follow python slice semantics, and traced/array indices are passed
    through to the leaves (jnp's out-of-bounds semantics apply to them).
    """
    batch = tree_shape(tree)
    if batch == ():
        raise TypeError(
            f"{type(tree).__name__} is not batched (batch shape ()): "
            "there are no batch dimensions to index into"
        )
    if isinstance(index, tuple):
        if len(index) > len(batch):
            raise IndexError(
                f"too many indices for batch shape {batch}: {index!r}"
            )
        for axis, subindex in enumerate(index):
            _check_bounds(tree, subindex, batch, axis)
    else:
        _check_bounds(tree, index, batch, axis=0)
    return jax.tree.map(lambda x: x[index], tree)


def _check_bounds(tree, index, batch, axis):
    if isinstance(index, int) and not -batch[axis] <= index < batch[axis]:
        raise IndexError(
            f"index {index} out of bounds for batch axis {axis} of "
            f"{type(tree).__name__} with batch shape {batch}"
        )


def tree_len(tree) -> int:
    """
    The leading batch dimension of a struct. Raises TypeError for an
    unbatched struct (batch shape ()).
    """
    batch = tree_shape(tree)
    if batch == ():
        raise TypeError(
            f"{type(tree).__name__} is not batched (batch shape ()): "
            "it has no length"
        )
    return batch[0]


def tree_iter(tree):
    """
    Iterate over the leading batch dimension of a struct, yielding structs
    with that dimension indexed away. Raises TypeError for an unbatched
    struct.
    """
    for i in range(tree_len(tree)):
        yield tree_getitem(tree, i)


def tree_size(tree) -> int:
    """Calculates the total number of parameters in the PyTree."""
    return sum(jnp.size(x) for x in jax.tree.leaves(tree))
