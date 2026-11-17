"""The public decorator preserves values while checking recursive descriptions."""

import functools
import inspect
import subprocess
import sys
from typing import Annotated, Self, TypeVar, no_type_check

import jax
import jax.numpy as jnp
import pytest
import jaxtyping
from beartype import beartype
from beartype.vale import Is
from jaxtyping import Array, Float, TypeCheckError, jaxtyped

import strux


@strux.struct
class Layer:
    weights: Float[Array, "out in"]
    biases: Float[Array, "out"]


@strux.struct
class Network:
    first: Layer
    second: Layer


def layer():
    return Layer(jnp.ones((3, 2)), jnp.zeros(3))


@pytest.mark.parametrize(
    "decorator", [strux.checked, strux.checked(), strux.checked(typechecker=beartype)]
)
def test_default_schema_on_arguments_and_returns(decorator):
    calls = []

    @decorator
    def keep(value: Network) -> Network:
        calls.append(value)
        return value

    good = Network(layer(), Layer(jnp.ones((2, 3)), jnp.zeros(2)))
    bad = good.replace(second=good.second.replace(biases=jnp.ones(4)))
    assert keep(good) is good
    with pytest.raises(TypeCheckError, match="parameters"):
        keep(bad)
    assert len(calls) == 1

    @decorator
    def bad_return(value: Network) -> Network:
        return bad

    with pytest.raises(TypeCheckError, match="return value"):
        bad_return(good)


def test_constructor_replace_and_nominal_escape_remain_polymorphic():
    malformed = Layer("weights", False).replace(biases=42)

    @strux.checked
    def keep(value: strux.InstanceOf[Layer]) -> strux.InstanceOf[Layer]:
        return value

    assert keep(malformed) is malformed
    assert isinstance(malformed, strux.InstanceOf[Layer])
    malformed.check(strux.InstanceOf[Layer])
    with pytest.raises(strux.ValidationError):
        malformed.check(Layer)
    with pytest.raises(TypeCheckError):
        keep(42)
    with pytest.raises(strux.ValidationError):
        strux.check(42, strux.InstanceOf[Layer])


def test_instanceof_in_fields_and_leaf_transformations():
    @strux.struct
    class Holder:
        item: strux.InstanceOf[Layer]

    value = Holder(Layer(True, False))
    value.check()
    with pytest.raises(strux.ValidationError):
        value.replace(item=42).check()
    with pytest.raises(strux.SchemaError, match="no leaf annotations"):
        strux.Struct[Holder, "n"]


def test_class_objects_are_not_translated_into_instance_descriptions():
    @strux.struct
    class Child(Layer):
        pass

    @strux.checked
    def keep(cls: type[Layer]) -> type[Layer]:
        return cls

    assert keep(Layer) is Layer
    assert keep(Child) is Child
    with pytest.raises(TypeCheckError):
        keep(Network)
    with pytest.raises(TypeCheckError):
        keep(layer())


def test_containers_unions_and_annotated_validators():
    positive = Is[lambda value: bool(jnp.all(value.weights > 0))]

    @strux.checked
    def keep(
        values: tuple[Layer, list[Layer | None], dict[str, Annotated[Layer, positive]]],
    ) -> tuple[Layer, list[Layer | None], dict[str, Layer]]:
        return values

    good = layer()
    assert keep((good, [None], {"one": good}))[0] is good
    bad = good.replace(biases=jnp.zeros(4))
    for values in (
        (bad, [None], {"one": good}),
        (good, [bad], {"one": good}),
        (good, [None], {"one": bad}),
        (good, [None], {"one": good.replace(weights=-good.weights)}),
    ):
        with pytest.raises(TypeCheckError):
            keep(values)


def test_original_annotations_signature_and_wrapped_function_are_preserved():
    def original(value: Layer, /, *, count: int = 1) -> Layer:
        """A function with a public signature."""
        return value

    annotations = dict(original.__annotations__)
    signature = inspect.signature(original)
    wrapped = strux.checked(original)
    good = layer()
    assert wrapped(good, count=2) is good
    assert original.__annotations__ == annotations
    assert wrapped.__annotations__ == annotations
    assert inspect.signature(wrapped) == signature
    assert inspect.unwrap(wrapped) is original
    assert wrapped.__doc__ == original.__doc__
    with pytest.raises(TypeCheckError):
        wrapped(good, count="two")


def test_explicit_backend_is_used_and_compiled_once():
    decorated = []

    def backend(fn):
        decorated.append(fn)
        return beartype(fn)

    @strux.checked(typechecker=backend)
    def keep(value: Layer) -> Layer:
        return value

    value = layer()
    assert keep(value) is value
    count = len(decorated)
    assert count > 0
    assert keep(value) is value
    assert len(decorated) == count


def test_forward_references_and_self_in_local_methods():
    @strux.struct
    class Local:
        values: Float[Array, "2"]

        @strux.checked
        def keep(self: "Local", other: "Local") -> Self:
            return other

        @strux.checked
        def wrong(self: Self) -> Self:
            return self.replace(values=jnp.ones(3))

        @strux.checked
        @classmethod
        def make(cls: type[Self]) -> Self:
            return cls(jnp.ones(2))

        @classmethod
        @strux.checked
        def make_other(cls: type[Self]) -> Self:
            return cls(jnp.ones(2))

        @strux.checked
        @staticmethod
        def identity(value: Layer) -> Layer:
            return value

        @strux.checked
        @property
        def copy(self: Self) -> Self:
            return self

    value = Local.make()
    assert value.keep(value) is value
    assert value.copy is value
    assert isinstance(Local.make_other(), Local)
    with pytest.raises(TypeCheckError):
        value.keep(value.replace(values=jnp.ones(3)))
    with pytest.raises(TypeCheckError):
        value.replace(values=jnp.ones(3)).keep(value)
    with pytest.raises(TypeCheckError):
        value.wrong()
    with pytest.raises(TypeCheckError):
        Local.identity(layer().replace(biases=jnp.ones(4)))

    @strux.struct
    class Child(Local):
        pass

    assert type(Child.make()) is Child
    child = Child.make()
    assert child.keep(child) is child
    with pytest.raises(TypeCheckError):
        child.keep(
            value
        )  # Self return requires Child, even though argument permits Local.


def test_unresolved_forward_reference_fails_instead_of_becoming_any():
    @strux.checked
    def keep(value: "MissingTypeForCheckedTest"):
        return value

    with pytest.raises(TypeError, match="could not resolve annotations"):
        keep(42)


def test_local_string_annotations():
    @strux.struct
    class Local:
        value: Float[Array, "2"]

    @strux.checked
    def keep(value: "Local") -> "Local":
        return value

    good = Local(jnp.ones(2))
    assert keep(good) is good
    with pytest.raises(TypeCheckError):
        keep(Local(jnp.ones(3)))


def test_type_aliases_and_typevar_bounds_do_not_bypass_field_checks():
    type Alias = Layer
    type Pair[T] = tuple[T, T]
    T = TypeVar("T", bound=Layer)
    C = TypeVar("C", Layer, int)

    @strux.checked
    def aliases(value: Pair[Alias]) -> Pair[Alias]:
        return value

    @strux.checked
    def bounded(value: T) -> T:
        return value

    @strux.checked
    def constrained(value: C) -> C:
        return value

    good = layer()
    bad = good.replace(biases=jnp.ones(4))
    assert aliases((good, good))[0] is good
    assert bounded(good) is good
    assert constrained(good) is good
    assert constrained(42) == 42
    for function, value in ((aliases, (good, bad)), (bounded, bad), (constrained, bad)):
        with pytest.raises(TypeCheckError):
            function(value)


def test_transformed_descriptions_jit_vmap_and_shared_return_dimensions():
    scalar_layer = strux.Struct[Layer, strux.Map(lambda _: Float[Array, ""])]
    summaries = strux.Batch[scalar_layer, "members"]

    @strux.checked
    def summarise(value: Layer) -> scalar_layer:
        return jax.tree.map(jnp.sum, value)

    @jax.jit
    @strux.checked
    def summarise_many(values: strux.Batch[Layer, "members"]) -> summaries:
        return jax.vmap(summarise)(values)

    values = Layer(jnp.ones((4, 3, 2)), jnp.ones((4, 3)))
    result = summarise_many(values)
    assert result.weights.shape == (4,)
    assert jnp.array_equal(result.weights, jnp.full(4, 6.0))

    @strux.checked
    def truncate(values: strux.Struct[Layer, "members"]) -> summaries:
        return jax.tree.map(lambda x: x[:-1], summarise_many(values))

    with pytest.raises(TypeCheckError, match="members=4"):
        truncate(values)


def test_scan_and_tree_map_preserve_classes_without_checking_reconstruction():
    @strux.struct
    class State:
        value: Float[Array, ""]

    @strux.checked
    def step(state: State, increment: Float[Array, ""]) -> tuple[State, State]:
        updated = state.replace(value=state.value + increment)
        return updated, updated

    @jax.jit
    @strux.checked
    def rollout(
        state: State, increments: Float[Array, "time"]
    ) -> tuple[State, strux.Struct[State, "time"]]:
        return jax.lax.scan(step, state, increments)

    final, trajectory = rollout(State(jnp.array(0.0)), jnp.ones(3))
    assert type(trajectory) is State
    assert final.value == 3
    assert jnp.array_equal(trajectory.value, jnp.array([1.0, 2.0, 3.0]))
    mask = jax.tree.map(lambda x: x > 0, trajectory)
    mask.check(strux.InstanceOf[State])


def test_plain_jaxtyped_and_explicit_check_remain_compatible():
    @jaxtyped(typechecker=beartype)
    def keep(value: strux.InstanceOf[Layer]) -> strux.Struct[Layer, ""]:
        return value

    assert keep(layer()).weights.shape == (3, 2)
    with pytest.raises(TypeCheckError, match="return value"):
        keep(Layer(True, False))


def test_no_backend_means_context_without_automatic_checks():
    @strux.checked(typechecker=None)
    def keep(value: Layer) -> Layer:
        return value

    assert keep(42) == 42

    @strux.checked(typechecker=None)
    def unresolved(value: "MissingTypeForDisabledTest"):
        return value

    assert unresolved(42) == 42

    @strux.struct
    class Local:
        value: Float[Array, "2"]

        @strux.checked(typechecker=None)
        def keep(self: "Local") -> Self:
            return self

    malformed = Local(False)
    assert malformed.keep() is malformed


def test_optional_backend_dependency_is_lazy():
    program = """
import sys
class BlockBeartype:
    def find_spec(self, fullname, *args):
        if fullname == "beartype" or fullname.startswith("beartype."):
            raise ImportError("beartype deliberately unavailable")
sys.meta_path.insert(0, BlockBeartype())
import strux
@strux.struct
class Item:
    value: float
Item(1.0).check()
try:
    @strux.checked
    def f(value: Item):
        return value
except ImportError as error:
    assert "strux[beartype]" in str(error)
else:
    raise AssertionError("default backend must require the extra")
@strux.checked(typechecker=lambda fn: fn)
def g(value: Item):
    return value
assert g(Item(1.0)).value == 1.0
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_standard_disabling_controls_apply_before_annotation_resolution(monkeypatch):
    @no_type_check
    @strux.checked
    def outside(value: Layer) -> Layer:
        return value

    @strux.checked
    @no_type_check
    def inside(value: Layer) -> Layer:
        return value

    @strux.checked
    def unresolved(value: "MissingTypeForGlobalDisable"):
        return value

    assert outside(42) == 42
    assert inside(42) == 42
    monkeypatch.setattr(jaxtyping.config, "jaxtyping_disable", True)
    assert unresolved(42) == 42


def test_rejects_unsupported_decorator_targets():
    async def asynchronous(value: Layer):
        return value

    def generator(value: Layer):
        yield value

    for target in (asynchronous, generator, Layer):
        with pytest.raises(TypeError):
            strux.checked(target)


def test_repeated_checking_replaces_configuration_and_preserves_opt_outs():
    @strux.checked(typechecker=None)
    @strux.checked
    def unchecked(value: Layer) -> Layer:
        return value

    assert unchecked(42) == 42
    assert not hasattr(unchecked.__wrapped__, "__wrapped__")

    @strux.checked
    @no_type_check
    @strux.checked
    def excluded(value: Layer) -> Layer:
        return value

    assert excluded(42) == 42


def test_repeated_checking_keeps_other_decorators_and_original_local_names():
    def make_function():
        LocalAlias = Layer

        @strux.checked
        def identity(value: "LocalAlias") -> "LocalAlias":
            return value

        return identity

    LocalAlias = int
    original = make_function()
    rechecked = strux.checked(original)
    assert rechecked(layer()).weights.shape == (3, 2)
    with pytest.raises(TypeCheckError):
        rechecked(42)

    @strux.checked
    def keep(value: Layer) -> Layer:
        return value

    @functools.wraps(keep)
    def change_result(*args, **kwargs):
        keep(*args, **kwargs)
        return Layer(True, False)

    # functools.wraps copies the private marker as well as annotations.
    # Checking must preserve this intervening decorator and check its output.
    with pytest.raises(TypeCheckError, match="return value"):
        strux.checked(change_result)(layer())
