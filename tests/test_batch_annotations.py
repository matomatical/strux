"""Uniform Batch descriptions preserve scopes and compose with Struct."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype import beartype
from jaxtyping import Array, Bool, Float, Shaped, TypeCheckError, jaxtyped

import strux


@strux.struct
class Layer:
    weights: Float[Array, "inputs outputs"]
    bias: Float[Array, "outputs"]


@strux.struct(static_fieldnames=("activation",))
class Network:
    first: Layer
    second: Layer
    activation: object = jax.nn.relu


def networks():
    return Network(
        first=Layer(jnp.ones((3, 2, 4)), jnp.zeros((3, 4))),
        second=Layer(jnp.ones((3, 4, 1)), jnp.zeros((3, 1))),
    )


@pytest.mark.parametrize("form", [strux.Batch, strux.Struct])
def test_batch_struct_descriptions_accept_and_reject_the_same_values(form):
    values = networks()
    description = form[Network, "members"]
    before = vars(values).copy()
    assert values.check(description) is None
    assert isinstance(values, description)
    assert vars(values) == before
    assert values.activation is jax.nn.relu
    wrong_batch = values.replace(second=jax.tree.map(lambda x: x[:2], values.second))
    wrong_dtype = jax.tree.map(lambda x: x > 0, values)
    for invalid in (wrong_batch, wrong_dtype):
        assert not isinstance(invalid, description)
        with pytest.raises(strux.ValidationError):
            invalid.check(description)


def test_batch_and_struct_can_alternate_without_changing_operation_order():
    scalar_leaves = strux.Map(lambda _: Float[Array, ""])
    summary = strux.Struct[Network, scalar_leaves]
    description = strux.Batch[summary, "members"]
    via_batch = strux.Struct[strux.Batch[Network, "members"], scalar_leaves]
    assert via_batch is strux.Struct[Network, "members", scalar_leaves]

    def summarise(net):
        return jax.tree.map(jnp.sum, net)

    per_member = jax.vmap(summarise)(networks())
    per_member.check(description)
    per_member.check(strux.Struct[summary, "members"])
    assert not isinstance(per_member, via_batch)
    summarise(networks()).check(via_batch)
    with pytest.raises(strux.ValidationError):
        summarise(networks()).check(description)


@pytest.mark.parametrize("element", [Layer, Float[Array, "2"]])
def test_nested_batch_prefixes_preserve_order_and_anonymous_axis_independence(element):
    description = strux.Batch[strux.Batch[element, "_"], "_"]
    assert description is strux.Batch[element, "_ _"]
    value = (
        Layer(jnp.ones((2, 3, 4, 5)), jnp.ones((2, 3, 5)))
        if element is Layer
        else jnp.ones((2, 3, 2))
    )
    strux.check(value, description)
    concrete = strux.Batch[strux.Batch[element, "3"], "2"]
    strux.check(value, concrete)
    assert not isinstance(value, strux.Batch[element, "3 2"])


def test_batch_preserves_element_scope_in_arrays_and_structs():
    @strux.struct
    class Vector:
        values: Float[Array, "n"]

    array = jnp.ones((3, 2))
    array_description = strux.Batch[Float[Array, "n"], "n"]
    strux.check(array, array_description)
    Vector(array).check(strux.Batch[Vector, "n"])
    assert not isinstance(array, Shaped[Float[Array, "n"], "n"])
    with jaxtyped("context"):
        assert isinstance(array, array_description)
        assert isinstance(jnp.ones(3), Float[Array, "n"])
        assert not isinstance(jnp.ones(2), Float[Array, "n"])


@pytest.mark.parametrize("form", [strux.Batch, strux.Struct])
def test_nested_field_batch_scope_is_distinct_from_its_element_scope(form):
    @strux.struct
    class Vector:
        values: Float[Array, "n"]
        offset: Float[Array, "n"]

    @strux.struct
    class Wrapper:
        vectors: form[Vector, "n"]

    value = Wrapper(Vector(jnp.ones((3, 2)), jnp.ones((3, 2))))
    value.check()
    with pytest.raises(strux.ValidationError):
        Wrapper(Vector(jnp.ones((3, 2)), jnp.ones((4, 2)))).check()
    with pytest.raises(strux.ValidationError):
        Wrapper(Vector(jnp.ones((3, 2)), jnp.ones((3, 4)))).check()


@pytest.mark.parametrize(
    "input_form, output_form",
    [(strux.Batch, strux.Struct), (strux.Struct, strux.Batch)],
)
def test_boundary_checking_connects_batch_arrays_and_structs_under_jit(
    input_form, output_form
):
    @jaxtyped(typechecker=beartype)
    def keep(
        values: input_form[Network, "members"],
        flags: strux.Batch[Bool[Array, ""], "members"],
    ) -> output_form[Network, "members"]:
        return values

    values = networks()
    result = jax.jit(keep)(values, jnp.ones(3, dtype=bool))
    np.testing.assert_array_equal(result.first.weights, values.first.weights)
    with pytest.raises(TypeCheckError):
        keep(values, jnp.ones(2, dtype=bool))

    @jaxtyped(typechecker=beartype)
    def shorten(
        values: input_form[Network, "members"],
    ) -> output_form[Network, "members"]:
        return jax.tree.map(lambda x: x[:-1], values)

    with pytest.raises(TypeCheckError):
        shorten(values)


def test_batch_descriptions_check_elements_inside_vmap():
    @jaxtyped(typechecker=beartype)
    def one(values: strux.Batch[Layer, ""]) -> strux.Batch[float, ""]:
        return jnp.sum(values.weights)

    @jaxtyped(typechecker=beartype)
    def many(
        values: strux.Batch[Layer, "members"],
    ) -> strux.Batch[float, "members"]:
        return jax.vmap(one)(values)

    np.testing.assert_array_equal(jax.jit(many)(networks().first), jnp.full(3, 8.0))


@pytest.mark.parametrize("form", [strux.Batch, strux.Struct])
def test_map_can_introduce_equivalent_batched_structs_with_shared_axes(form):
    @strux.struct
    class Statistic:
        value: Float[Array, ""]

    description = strux.Struct[Layer, strux.Map(lambda _: form[Statistic, "n"])]
    good = Layer(Statistic(jnp.ones(3)), Statistic(jnp.zeros(3)))
    bad = good.replace(bias=Statistic(jnp.ones(4)))
    good.check(description)
    assert not isinstance(bad, description)
    with jaxtyped("context"):
        assert isinstance(good, description)
        assert not isinstance(jnp.ones(4), Float[Array, "n"])


@pytest.mark.parametrize("form", [strux.Batch, strux.Struct])
def test_batched_transformed_fields_support_shape_indexing_and_restoration(
    form, tmp_path
):
    summary = strux.Struct[Layer, strux.Map(lambda _: Float[Array, ""])]

    @strux.struct
    class Report:
        layers: form[summary, "2"]
        score: Float[Array, ""]

    value = Report(Layer(jnp.ones((3, 2)), jnp.zeros((3, 2))), jnp.zeros(3))
    assert value.shape == (3,)
    assert value[0].shape == ()
    value.check(strux.Batch[Report, "3"])
    path = tmp_path / "report.npz"
    value.save(path)
    restored = strux.load(path, template=Report)
    restored.check(strux.Batch[Report, "3"])
    np.testing.assert_array_equal(restored.layers.weights, value.layers.weights)


def test_batch_can_describe_containers_and_scalar_shorthand():
    description = strux.Batch[tuple[float, int], "members"]
    strux.check((jnp.ones(3), jnp.arange(3)), description)
    assert not isinstance((jnp.ones(3), jnp.arange(2)), description)
    assert not isinstance((jnp.ones(3), jnp.ones(3)), description)
    assert isinstance(1.0, strux.Batch[float, ""])
    assert not isinstance(1.0, strux.Batch[float, "members"])


def test_array_batch_failures_do_not_bind_external_dimensions():
    description = strux.Batch[Float[Array, "2"], "n"]
    with jaxtyped("context"):
        assert not isinstance(jnp.ones((4, 3)), description)
        assert isinstance(jnp.ones((3, 2)), description)
        assert isinstance(jnp.ones(3), Float[Array, "n"])


def test_batch_accepts_only_a_dimension_string_and_struct_rejects_array_targets():
    for operation in (strux.Map(lambda _: int), strux.Replace(bias=int), 3):
        with pytest.raises(TypeError, match="dimensions must be a string"):
            strux.Batch[Layer, operation]
    for arguments in ((Layer,), (Layer, "n", "m")):
        with pytest.raises(
            TypeError, match="takes an annotation and a dimension string"
        ):
            strux.Batch[arguments]
    with pytest.raises(TypeError, match="dataclass"):
        strux.Struct[strux.Batch[Float[Array, "2"], "n"], strux.Map(lambda _: int)]
