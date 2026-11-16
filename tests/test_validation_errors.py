"""Diagnostics explain failures without changing checking or scope rollback."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float, Int, jaxtyped

import strux


@strux.struct
class Particle:
    position: Float[Array, "2"]
    velocity: Float[Array, "2"]
    mass: float


@pytest.mark.parametrize("array", [jnp.ones, np.ones])
def test_scalar_shorthand_reports_relevant_array_dimension_failure(array):
    value = Particle(jnp.ones((3, 2)), jnp.ones((3, 2)), array(2))
    with pytest.raises(strux.ValidationError) as caught:
        value.check(strux.Struct[Particle, "n"])
    assert str(caught.value) == (
        "Particle.mass: batch dimension 'n' has size 2; expected 3.\n"
        "  Expected shape: (n,)\n"
        "  Actual shape:   (2,)"
    )
    failure = caught.value._failure
    assert failure.expected == 3
    assert failure.actual == 2
    assert failure.axis.batch
    assert type(caught.value) is strux.ValidationError


@pytest.mark.parametrize(
    "value, message",
    [
        (jnp.ones(2, dtype=bool), "expected Float dtype; got bool"),
        (jnp.ones((1, 2)), "expected rank 1; got rank 2"),
        (jnp.ones(3), "dimension '2' has size 3; expected 2"),
        ([1.0, 2.0], "expected Array; got list"),
    ],
)
def test_common_array_mismatches(value, message):
    @strux.struct
    class Vector:
        values: Float[Array, "2"]

    with pytest.raises(strux.ValidationError) as caught:
        Vector(value).check()
    assert str(caught.value).startswith(f"Vector.values: {message}.")
    assert "struxlocal" not in str(caught.value)


def test_same_spelling_in_batch_and_element_scopes_is_explained_separately():
    @strux.struct
    class Vectors:
        first: Float[Array, "n"]
        second: Float[Array, "n"]

    description = strux.Struct[Vectors, "n"]
    for shape, expected in [
        ((3, 4), "dimension 'n' has size 4; expected 2"),
        ((4, 2), "batch dimension 'n' has size 4; expected 3"),
    ]:
        with pytest.raises(strux.ValidationError) as caught:
            Vectors(jnp.ones((3, 2)), jnp.ones(shape)).check(description)
        assert str(caught.value).startswith(f"Vectors.second: {expected}.")
    Vectors(jnp.ones((3, 2)), jnp.ones((3, 2))).check(description)


def test_repeated_dimension_in_one_leaf_uses_the_first_axis():
    @strux.struct
    class Square:
        matrix: Float[Array, "n n"]

    with pytest.raises(strux.ValidationError) as caught:
        Square(jnp.ones((3, 2))).check()
    assert "dimension 'n' has size 2; expected 3" in str(caught.value)


def test_genuine_union_retains_distinct_dtype_and_shape_alternatives():
    @strux.struct
    class Choice:
        values: Float[Array, "2"] | Int[Array, "3"]

    with pytest.raises(strux.ValidationError) as caught:
        Choice(jnp.ones(3)).check()
    assert str(caught.value).startswith("Choice.values: no alternative matches:")
    assert "dimension '2' has size 3; expected 2" in str(caught.value)
    assert "expected Int dtype; got float32" in str(caught.value)
    assert len(caught.value._failure.alternatives) == 2
    Choice(jnp.ones(2)).check()
    Choice(jnp.ones(3, dtype=int)).check()


def test_backtracking_keeps_the_failure_furthest_through_fields():
    @strux.struct
    class Choice:
        pair: (
            tuple[Float[Array, "n"], Float[Array, "m"]]
            | tuple[Float[Array, "m"], Float[Array, "n"]]
        )
        first: Float[Array, "n"]
        last: Float[Array, "m"]

    value = Choice((jnp.ones(2), jnp.ones(3)), jnp.ones(2), jnp.ones(4))
    with pytest.raises(strux.ValidationError) as caught:
        value.check()
    assert str(caught.value).startswith(
        "Choice.last: dimension 'm' has size 4; expected 3."
    )
    value.replace(last=jnp.ones(3)).check()
    value.replace(first=jnp.ones(3), last=jnp.ones(2)).check()


@pytest.mark.parametrize(
    "annotation, value, expected",
    [
        (tuple[float, int], [1.0, 2], "expected tuple; got list"),
        (tuple[float, int], (1.0,), "expected 2 tuple entries; got 1"),
        (dict[str, float], {1: 2.0}, "key 1 does not match"),
        (Particle, 3, "expected Particle; got int"),
        (None, 3, "expected None; got int"),
    ],
)
def test_structural_failures(annotation, value, expected):
    @strux.struct
    class Wrapper:
        contents: annotation

    with pytest.raises(strux.ValidationError) as caught:
        Wrapper(value).check()
    assert expected in str(caught.value)
    assert caught.value._failure.path == "Wrapper.contents"


def test_symbolic_shapes_use_a_readable_fallback():
    @strux.struct
    class Item:
        values: Float[Array, "n n+1"]

    with pytest.raises(strux.ValidationError) as caught:
        Item(jnp.ones((2, 4))).check()
    assert str(caught.value) == (
        "Item.values: shape does not match the annotation.\n"
        "  Expected shape: (n, n+1)\n"
        "  Actual shape:   (2, 4)"
    )
    Item(jnp.ones((2, 3))).check()


def test_separate_variadic_segments_have_a_readable_fallback():
    @strux.struct
    class Variable:
        values: Float[Array, "*features 2"]
        scalar: float

    annotation = strux.Struct[Variable, "*batch"]
    value = Variable(jnp.ones((3, 4, 5)), jnp.ones(3))
    with pytest.raises(strux.ValidationError) as caught:
        value.check(annotation)
    assert "Expected shape: (*batch, *features, 2)" in str(caught.value)
    assert "Actual shape:   (3, 4, 5)" in str(caught.value)
    assert "strux" not in str(caught.value)
    value.replace(values=jnp.ones((3, 4, 2))).check(annotation)


def test_diagnostic_probes_do_not_publish_failed_bindings_under_jit():
    description = strux.Batch[Float[Array, "n n"], "members"]

    @jax.jit
    def probe(bad, good):
        with jaxtyped("context"):
            assert not isinstance(bad, description)
            assert isinstance(good, description)
            assert isinstance(jnp.ones(3), Float[Array, "members"])
            # The element's n is local to the description.
            assert isinstance(jnp.ones(7), Float[Array, "n"])
        return good

    result = probe(jnp.ones((4, 2, 3)), jnp.ones((3, 2, 2)))
    assert result.shape == (3, 2, 2)
