"""
Shapes and indexing for (batched) structs: the batch shape, symbolic dim
binding, batch indexing, and total leaf count.
"""

import jax
import jax.numpy as jnp

from strux.schema import schema
from strux.batch import (
    _TOP,
    _bind_names,
    _format_candidates,
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
    The symbolic dimension names appearing in a struct's field annotations,
    bound to their sizes in this instance: annotate `weights: Float[Array,
    "n_in n_out"]` and `biases: Float[Array, "n_out"]` and `tree_dims(net)`
    returns `{"n_in": ..., "n_out": ...}`.

    Names are scoped to the class whose annotations mention them (checked
    at construction): nested struct fields bind their own names at their
    own level, so this query returns only the names in `type(tree)`'s own
    annotations — call it on a nested struct to read that struct's names.
    """
    # the batch shape is needed only to pick union arms; without a
    # determined batch, ambiguous unions are skipped rather than guessed
    try:
        batch = tree_shape(tree)
    except ValueError:
        batch = None
    bindings = _bind_names(tree, schema(type(tree)), batch)
    return {name: size for name, (size, _) in bindings.items()}


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
