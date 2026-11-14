"""
Shapes and indexing for (batched) structs: the batch shape, symbolic dim
binding, batch indexing, and total leaf count.
"""

import numbers
import operator

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

    Explicit hints take precedence. Otherwise every data field contributes
    a shape constraint; dtype changes do not prevent batch access. If
    variadic annotations leave more than one possible batch shape, supply
    .as_batch(shape=...) to choose the intended prefix. Empty trees default
    to shape (). Inference reads current fields without storing a cache.
    """
    hint = getattr(tree, "_strux_batch_shape", None)
    if hint is not None:
        return hint
    candidates = _solved_candidates(tree)
    if candidates is _TOP:
        return ()
    if len(candidates) == 1:
        return next(iter(candidates))
    raise ValueError(
        f"{type(tree).__name__}: batch shape under-determined: consistent "
        f"candidates {_format_candidates(candidates)}; annotate "
        "element dims or use .as_batch(shape=...) to choose the batch shape"
    )


def tree_dims(tree) -> dict:
    """
    The symbolic dimension names appearing in a struct's field annotations,
    bound to their sizes in this instance: annotate `weights: Float[Array,
    "n_in n_out"]` and `biases: Float[Array, "n_out"]` and `tree_dims(net)`
    returns `{"n_in": ..., "n_out": ...}`.

    Names are scoped to each occurrence of the declaring class:
    nested struct fields bind their own names at their
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
    indices = index if isinstance(index, tuple) else (index,)
    if sum(item is Ellipsis for item in indices) > 1:
        raise IndexError("only one ellipsis is allowed")
    consumed = sum(_index_axes(item) for item in indices)
    if consumed > len(batch):
        raise IndexError(f"too many indices for batch shape {batch}")
    expanded = []
    for item in indices:
        if item is Ellipsis:
            expanded.extend([slice(None)] * (len(batch) - consumed))
        else:
            expanded.append(item)
    if not any(item is Ellipsis for item in indices):
        expanded.extend([slice(None)] * (len(batch) - consumed))
    axis = 0
    for item in expanded:
        consumed_axes = _index_axes(item)
        if consumed_axes == 0:
            continue
        if (
            isinstance(item, numbers.Integral)
            and not -batch[axis] <= item < batch[axis]
        ):
            raise IndexError(
                f"index {item} out of bounds for batch axis {axis} of {type(tree).__name__} with batch shape {batch}"
            )
        axis += consumed_axes
    prefix_index = tuple(expanded)
    # Shape-only tracing avoids allocating an array the size of the batch.
    result_shape = jax.eval_shape(
        lambda placeholder: placeholder[prefix_index],
        jax.ShapeDtypeStruct(batch, jnp.int8),
    ).shape
    result = jax.tree.map(lambda leaf: leaf[prefix_index], tree)
    if getattr(tree, "_strux_batch_shape", None) is not None:
        result = as_batch(result, shape=result_shape)
    return result


def _index_axes(index):
    if index is None or index is Ellipsis:
        return 0
    if isinstance(index, bool) or getattr(index, "dtype", None) == jnp.bool_:
        return getattr(index, "ndim", 0)
    return 1


def as_batch(tree, *, shape):
    """Return a same-class copy with an explicit leading batch shape.

    Array data is shared. Hints govern batch access, do not validate an
    annotation, and are discarded by generic JAX tree reconstruction and
    dataclass replacement. Nested structs receive the same prefix hint.
    """
    if not hasattr(type(tree), "_data_fields"):
        raise TypeError("as_batch expects a strux struct")
    shape = tuple(shape)
    if any(isinstance(size, bool) for size in shape):
        raise TypeError("batch sizes must be integers, not booleans")
    try:
        shape = tuple(operator.index(size) for size in shape)
    except TypeError:
        raise TypeError("batch shape must contain concrete integer sizes") from None
    if any(size < 0 for size in shape):
        raise ValueError("batch dimensions must be nonnegative")
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        leaf_shape = getattr(leaf, "shape", ())
        if tuple(leaf_shape[: len(shape)]) != shape:
            raise ValueError(
                f"{jax.tree_util.keystr(path)}: shape {leaf_shape} does not have batch prefix {shape}"
            )

    def clone(node):
        children, definition = jax.tree_util.tree_flatten(
            node,
            is_leaf=lambda child: child is not node,
        )
        if definition.num_nodes == 1 and definition.num_leaves == 1:
            return node
        copied = jax.tree.unflatten(definition, [clone(child) for child in children])
        if hasattr(type(copied), "_data_fields"):
            object.__setattr__(copied, "_strux_batch_shape", shape)
        return copied

    return clone(tree)


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
