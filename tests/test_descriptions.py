"""Annotation descriptions, independent checking and native JAX integration."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from beartype import beartype
from jaxtyping import Array, Bool, Float, Int, TypeCheckError, jaxtyped

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


def network():
    return Network(
        first=Layer(jnp.ones((2, 3)), jnp.zeros(3)),
        second=Layer(jnp.ones((3, 1)), jnp.zeros(1)),
    )


def test_unchecked_construction_and_replace_then_explicit_check():
    layer = Layer(jnp.ones(2, bool), jnp.zeros(3, bool))
    with pytest.raises(strux.ValidationError, match="Layer.weights"):
        layer.check()
    mask = strux.Struct[
        Layer,
        strux.Replace(
            weights=Bool[Array, "inputs"],
            bias=Bool[Array, "outputs"],
        ),
    ]
    assert layer.check(mask) is None
    assert layer.replace(weights=jnp.zeros(4, bool)).check(mask) is None


def test_nested_element_dimensions_are_local_but_batches_are_shared():
    net = network()
    net.check()
    batched = jax.tree.map(lambda x: jnp.stack([x] * 4), net)
    batched.check(strux.Struct[Network, "n"])
    broken = batched.replace(second=jax.tree.map(lambda x: x[:2], batched.second))
    for dims in ("n", "...", "_"):
        assert not isinstance(broken, strux.Struct[Network, dims])


def test_failed_checks_do_not_pollute_boundary_dimensions():
    annotation = strux.Struct[Layer, "n"]
    good = Layer(jnp.ones((4, 2, 3)), jnp.ones((4, 3)))
    bad = Layer(jnp.ones((2, 2, 3)), jnp.ones((3, 3)))
    with jaxtyped("context"):
        assert not isinstance(bad, annotation)
        assert isinstance(good, annotation)
        assert isinstance(jnp.ones(4), Float[Array, "n"])


def test_union_failure_rolls_back_local_dimensions():
    @strux.struct
    class Choice:
        arrays: (
            tuple[Float[Array, "n"], Int[Array, "n"]]
            | tuple[Float[Array, "m"], Float[Array, "m"]]
        )
        other: Float[Array, "n"]

    Choice((jnp.ones(3), jnp.ones(3)), jnp.ones(4)).check()


def test_map_introduces_subtree_then_next_map_visits_it():
    @strux.struct
    class Stats:
        mean: Float[Array, ""]
        maximum: Float[Array, ""]

    rich = strux.Struct[Network, strux.Map(lambda _: Stats)]
    flags = strux.Struct[rich, strux.Map(lambda _: Bool[Array, ""])]
    values = jax.tree.map(lambda x: Stats(x.mean(), x.max()), network())
    values.check(rich)
    bools = jax.tree.map(lambda x: x > 0, values)
    bools.check(flags)
    assert type(bools.first.weights) is Stats
    assert bools.activation is jax.nn.relu


def test_formal_batch_matches_shortcut_and_preserves_names():
    mapped = strux.Struct[Network, strux.Map(lambda a: strux.Batch[a, "n"])]
    values = jax.tree.map(lambda x: jnp.stack([x] * 4), network())
    values.check(mapped)
    values.check(strux.Struct[Network, "n"])
    with jaxtyped("context"):
        assert isinstance(values, mapped)
        assert not isinstance(jnp.ones(2), Float[Array, "n"])


def test_named_map_results_keep_element_names_local():
    description = strux.Struct[
        Network, "batch", strux.Map(lambda _: Float[Array, "summary"])
    ]
    values = Network(Layer(jnp.ones(3), jnp.ones(3)), Layer(jnp.ones(5), jnp.ones(5)))
    values.check(description)
    with jaxtyped("context"):
        assert isinstance(values, description)
        assert isinstance(jnp.ones(9), Float[Array, "summary"])


def test_nested_transformed_fields_shape_check_and_checkpoint(tmp_path):
    StatsDescription = strux.Struct[Network, strux.Map(lambda _: Float[Array, "2"])]

    @strux.struct
    class Report:
        stats: StatsDescription

    value = Report(jax.tree.map(lambda x: jnp.stack([x.mean(), x.max()]), network()))
    value.check()
    assert value.shape == ()
    path = tmp_path / "stats.npz"
    value.save(path)
    restored = strux.load(
        path, template=Report, statics={"stats/activation": jax.nn.relu}
    )
    restored.check()
    assert jax.tree.all(jax.tree.map(jnp.array_equal, value, restored))


def test_boundary_checks_work_inside_native_vmap_and_jit():
    summaries = strux.Struct[Layer, strux.Map(lambda _: Float[Array, ""])]

    @jaxtyped(typechecker=beartype)
    def summarise(layer: strux.Struct[Layer, ""]) -> summaries:
        return jax.tree.map(jnp.sum, layer)

    @jaxtyped(typechecker=beartype)
    def many(layers: strux.Struct[Layer, "n"]) -> strux.Struct[summaries, "n"]:
        return jax.vmap(summarise)(layers)

    layer = network().first
    layers = jax.tree.map(lambda x: jnp.stack([x] * 3), layer)
    result = jax.jit(many)(layers)
    assert np.array_equal(result.weights, np.full(3, 6))
    assert type(result) is Layer

    @jaxtyped(typechecker=beartype)
    def shortened(layers: strux.Struct[Layer, "n"]) -> strux.Struct[summaries, "n"]:
        return jax.tree.map(lambda x: x[:-1], many(layers))

    with pytest.raises(TypeCheckError):
        shortened(layers)


def test_bare_array_annotation_is_rejected_when_description_is_needed():
    @strux.struct
    class Bare:
        value: jax.Array

    value = Bare(jnp.ones(3))
    with pytest.raises(strux.SchemaError, match="jaxtyping"):
        value.check()


def test_unknown_field_and_invalid_map_result_are_declaration_errors():
    with pytest.raises(strux.SchemaError, match="unknown data field"):
        strux.Struct[Layer, strux.Replace(missing=Float[Array, ""])]
    with pytest.raises(strux.SchemaError):
        strux.Struct[Layer, strux.Map(lambda _: object)]


def test_union_alternative_can_be_selected_by_later_fields():
    @strux.struct
    class Choice:
        first: Float[Array, "n"] | Float[Array, "m"]
        second: Float[Array, "n"]

    Choice(jnp.ones(3), jnp.ones(4)).check()


def test_variadic_batch_and_element_shapes_backtrack_across_fields():
    @strux.struct
    class Variable:
        unknown: Float[Array, "..."]
        scalar: Float[Array, ""]

    value = Variable(jnp.ones((3, 2)), jnp.ones(3))
    value.check(strux.Struct[Variable, "..."])
    with pytest.raises(strux.ValidationError):
        value.replace(scalar=jnp.ones(4)).check(strux.Struct[Variable, "..."])


def test_annotation_check_does_not_mutate_value_or_bind_explicit_check_to_caller():
    layer = Layer(jnp.ones((4, 2, 3)), jnp.ones((4, 3)))
    original = dict(vars(layer))
    with jaxtyped("context"):
        assert isinstance(jnp.ones(2), Float[Array, "n"])
        layer.check(strux.Struct[Layer, "n"])
        assert isinstance(jnp.ones(2), Float[Array, "n"])
    assert vars(layer) == original


def test_element_names_do_not_collide_with_outer_batch_names():
    @strux.struct
    class SameName:
        values: Float[Array, "n"]

    SameName(jnp.ones((3, 5))).check(strux.Struct[SameName, "n"])


def test_pure_mapping_callback_does_not_run_again_for_each_value():
    calls = []

    def transform(annotation):
        calls.append(annotation)
        return strux.Batch[annotation, "n"]

    description = strux.Struct[Layer, strux.Map(transform)]
    value = Layer(jnp.ones((3, 2, 4)), jnp.ones((3, 4)))
    value.check(description)
    count = len(calls)
    value.check(description)
    assert len(calls) == count


def test_public_dimension_names_cannot_collide_with_internal_scopes():
    @strux.struct
    class Item:
        values: Float[Array, "n"]

    annotation = strux.Struct[Item, "struxlocal0_n"]
    with jaxtyped("context"):
        assert isinstance(Item(jnp.ones((3, 5))), annotation)
        assert isinstance(jnp.ones(3), Float[Array, "struxlocal0_n"])
        assert not isinstance(jnp.ones(5), Float[Array, "struxlocal0_n"])


def test_repeated_checks_reuse_jaxtyping_annotation_types():
    from jaxtyping._array_types import _make_array_cached

    value = network()
    value.check()
    size = _make_array_cached.cache_info().currsize
    for _ in range(10):
        value.check()
    assert _make_array_cached.cache_info().currsize == size


def test_element_expressions_and_named_anonymous_axes():
    @strux.struct
    class Item:
        values: Float[Array, "n n+1 _features"]
        other: Float[Array, "_features"]

    Item(jnp.ones((2, 3, 4)), jnp.ones(5)).check()
    with pytest.raises(strux.ValidationError):
        Item(jnp.ones((2, 4, 4)), jnp.ones(5)).check()


def test_dictionary_keys_are_checked_and_maps_preserve_their_annotation():
    @strux.struct
    class Lookup:
        values: dict[str, Float[Array, "2"]]

    Lookup({"ok": jnp.ones(2)}).check()
    with pytest.raises(strux.ValidationError, match="key"):
        Lookup({1: jnp.ones(2)}).check()
    masked = strux.Struct[Lookup, strux.Map(lambda _: Bool[Array, "2"])]
    with pytest.raises(strux.ValidationError, match="key"):
        Lookup({1: jnp.ones(2, bool)}).check(masked)


def test_batch_prefixes_normalise_in_prepend_order():
    assert strux.Struct[Layer, "members", "time"] is strux.Struct[Layer, "time members"]


def test_numpy_and_jax_array_requirements_are_preserved():
    @strux.struct
    class Host:
        values: Float[np.ndarray, "2"]

    Host(np.ones(2)).check()
    with pytest.raises(strux.ValidationError):
        Host(jnp.ones(2)).check()
    batched = strux.Struct[Host, strux.Map(lambda a: strux.Batch[a, "n"])]
    Host(np.ones((3, 2))).check(batched)
    assert not isinstance(Host(jnp.ones((3, 2))), batched)


def test_inserted_struct_statics_survive_class_template_restoration(tmp_path):
    @strux.struct(static_fieldnames=("label",))
    class Statistic:
        value: Float[Array, ""]
        label: str

    Description = strux.Struct[Layer, strux.Map(lambda _: Statistic)]

    @strux.struct
    class Report:
        layer: Description

    value = Report(
        jax.tree.map(lambda x: Statistic(x.mean(), "average"), network().first)
    )
    path = tmp_path / "rich.npz"
    value.save(path)
    restored = strux.load(path, template=Report)
    restored.check()
    assert restored.layer.weights.label == "average"
    assert isinstance(restored.layer.weights, Statistic)


def test_wide_containers_do_not_use_one_python_stack_frame_per_leaf():
    @strux.struct
    class Many:
        arrays: list[Float[Array, "n"]]

    Many([jnp.ones(2)] * 1500).check()


def test_formal_batch_accepts_scalar_shorthand_as_a_field_annotation():
    @strux.struct
    class Scalars:
        values: strux.Batch[float, "items"]

    Scalars(jnp.ones(3)).check()
    with pytest.raises(strux.ValidationError):
        Scalars(1.0).check()


def test_map_can_insert_batched_structs_then_map_their_leaves():
    @strux.struct
    class Stats:
        value: Float[Array, ""]

    batch = strux.Struct[Layer, strux.Map(lambda _: strux.Batch[Stats, "n"])]
    flags = strux.Struct[batch, strux.Map(lambda _: Bool[Array, ""])]
    value = Layer(Stats(jnp.ones(3)), Stats(jnp.ones(3)))
    value.check(batch)
    with jaxtyped("context"):
        assert isinstance(value, batch)
        assert not isinstance(jnp.ones(4), Float[Array, "n"])
    Layer(Stats(jnp.array(True)), Stats(jnp.array(False))).check(flags)
