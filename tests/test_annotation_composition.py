"""Named struct descriptions compose without changing value containers."""

import jax
import jax.numpy as jnp
import pytest
from beartype import beartype
from jaxtyping import Array, Float, TypeCheckError, jaxtyped

import strux

from example_structs import Point


def test_nested_batch_descriptions_preserve_prefix_order():
    four = strux.Struct[Point, "4"]
    three_by_four = strux.Struct[four, "3"]
    two_by_three_by_four = strux.Struct[three_by_four, "2"]

    assert two_by_three_by_four is strux.Struct[Point, "4", "3", "2"]
    assert isinstance(
        Point(x=jnp.zeros((2, 3, 4)), y=jnp.ones((2, 3, 4))),
        two_by_three_by_four,
    )
    assert not isinstance(
        Point(x=jnp.zeros((4, 3, 2)), y=jnp.ones((4, 3, 2))),
        two_by_three_by_four,
    )


def test_reduction_after_batching_differs_from_batched_reduction():
    batched_points = strux.Struct[Point, "n"]
    scalar_mask = strux.Struct[batched_points, strux.mapped(bool)]
    point_mask = strux.Struct[Point, strux.mapped(bool)]
    batched_mask = strux.Struct[point_mask, "n"]
    points = Point(x=jnp.arange(3.0), y=jnp.ones(3))

    reduced = jax.tree.map(lambda x: jnp.all(x > 0), points)
    compared = jax.tree.map(lambda x: x > 0, points)

    assert type(reduced) is Point
    assert type(compared) is Point
    assert isinstance(reduced, scalar_mask)
    assert not isinstance(reduced, batched_mask)
    assert isinstance(compared, batched_mask)
    assert not isinstance(compared, scalar_mask)


def test_named_summary_composes_with_checked_vmap_and_jit():
    @strux.struct
    class Measurement:
        values: Float[Array, "2"]
        score: Float[Array, ""]

    summary = strux.Struct[Measurement, strux.mapped(float)]
    summaries = strux.Struct[summary, "n"]

    @jaxtyped(typechecker=beartype)
    def summarise(value: strux.Struct[Measurement, ""]) -> summary:
        return jax.tree.map(jnp.sum, value)

    @jaxtyped(typechecker=beartype)
    def summarise_many(values: strux.Struct[Measurement, "n"]) -> summaries:
        return jax.vmap(summarise)(values)

    values = Measurement(values=jnp.arange(6.0).reshape(3, 2), score=jnp.ones(3))
    result = jax.jit(summarise_many)(values)

    assert type(result) is Measurement
    assert jnp.array_equal(result.values, jnp.array([1.0, 5.0, 9.0]))
    assert jnp.array_equal(result.score, jnp.ones(3))
    assert isinstance(result, summaries)
    assert not isinstance(result, strux.Struct[Measurement, "n"])


def test_composed_return_description_checks_shared_batch_dimension():
    mask = strux.Struct[Point, strux.astype(bool)]
    masks = strux.Struct[mask, "n"]

    @jaxtyped(typechecker=beartype)
    def shortened(points: strux.Struct[Point, "n"]) -> masks:
        return jax.tree.map(lambda x: x[:-1] > 0, points)

    with pytest.raises(TypeCheckError):
        shortened(Point(x=jnp.ones(3), y=jnp.ones(3)))


def test_composed_stacked_field_discounts_all_declared_axes():
    @strux.struct
    class Block:
        scale: Float[Array, "4"]

    depth = strux.Struct[Block, "3"]
    heads_and_depth = strux.Struct[depth, "2"]

    @strux.struct
    class Network:
        blocks: heads_and_depth
        score: Float[Array, ""]

    network = Network(blocks=Block(scale=jnp.ones((2, 3, 4))), score=jnp.array(1.0))

    assert network.shape == ()
    assert isinstance(network, strux.Struct[Network, ""])
    assert strux.schema(Network).fields["blocks"].dims == ("2", "3")


def test_composed_generic_description_retains_original_class():
    @strux.struct
    class Box[T: Point]:
        item: T

    mask = strux.Struct[Box[Point], strux.astype(bool)]
    batched_mask = strux.Struct[mask, "n"]
    boxes = Box(item=Point(x=jnp.ones(3), y=jnp.zeros(3)))
    compared = jax.tree.map(lambda x: x > 0, boxes)

    assert type(compared) is Box
    assert type(compared.item) is Point
    assert isinstance(compared, batched_mask)
    assert batched_mask is strux.Struct[Box[Point], strux.astype(bool), "n"]


def test_composition_rejects_unsupported_transformations():
    annotation = strux.Struct[Point, "n"]
    with pytest.raises(TypeError, match="dims strings or functors"):
        strux.Struct[annotation, 42]
