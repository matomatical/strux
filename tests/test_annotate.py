"""
Tests for batch annotations (Cls["batch"]): static expansion, runtime
isinstance checks, and integration with jaxtyping + beartype.
"""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Bool, Float, Int, jaxtyped
from beartype import beartype

import strux

from example_structs import Environment, Point, World


# # #
# Annotation expansion (static behaviour)


class TestAnnotationExpansion:
    def test_jaxtyping_fields_prepended(self):
        ann = Environment["batch"]
        hints = ann._field_hints
        assert hints["hero_pos"].dim_str == "batch 2"
        assert hints["goal_pos"].dim_str == "batch 2"
        assert hints["walls"].dim_str == "batch h w"

    def test_preserves_dtype_class(self):
        ann = Environment["batch"]
        hints = ann._field_hints
        # dtype should point back to the original jaxtyping dtype class
        assert hints["hero_pos"].dtype is Int
        assert hints["walls"].dtype is Bool

    def test_preserves_array_type(self):
        ann = Environment["batch"]
        hints = ann._field_hints
        for hint in hints.values():
            assert hint.array_type is jax.Array

    def test_nested_struct_recursion(self):
        ann = World["batch"]
        hints = ann._field_hints
        # env field should be an expanded struct annotation
        env_ann = hints["env"]
        assert env_ann._struct_type is Environment
        assert env_ann._field_hints["hero_pos"].dim_str == "batch 2"
        assert env_ann._field_hints["walls"].dim_str == "batch h w"
        # score field should be directly expanded
        assert hints["score"].dim_str == "batch"

    def test_meta_fields_skipped(self):
        @strux.struct(static_fieldnames=("name",))
        class WithMeta:
            pos: Int[Array, "2"]
            name: str
        ann = WithMeta["batch"]
        hints = ann._field_hints
        # jaxtyping data field is expanded
        assert hints["pos"].dim_str == "batch 2"
        # meta field is not included
        assert "name" not in hints

    def test_non_pytree_data_field_raises(self):
        @strux.struct
        class Bad:
            pos: Int[Array, "2"]
            name: str

        with pytest.raises(strux.SchemaError, match="not pytree data"):
            Bad["batch"]

    def test_jaxtype_detection_is_exact(self):
        # a class that merely duck-types some jaxtyping attributes is not
        # treated as an array annotation: it is an instance-checked pytree
        # class, and an instance (with no array leaves inside) fails
        # validation at construction
        class Impostor:
            dtype = None
            array_type = None
            dim_str = "2"

        @strux.struct
        class HasImpostor:
            pos: Impostor

        with pytest.raises(strux.ValidationError, match="not an array"):
            HasImpostor(pos=Impostor())

    def test_plain_scalar_hints_promoted(self):
        # plain float/int/bool hints mean "python scalar or array of the
        # matching dtype kind"; batching keeps the array arms
        @strux.struct
        class Metrics:
            pos: Int[Array, "2"]
            loss: float
            step: int
            done: bool

        ann = Metrics["batch"]
        # each scalar field expands to a union of batched array arms
        for name in ("loss", "step", "done"):
            arms = ann._field_hints[name]
            assert isinstance(arms, tuple)
            assert any(getattr(arm, "dim_str", None) == "batch" for arm in arms)
        # a batched instance passes the isinstance check
        batched = Metrics(
            pos=jnp.zeros((4, 2), dtype=jnp.int32),
            loss=jnp.zeros(4),
            step=jnp.zeros(4, dtype=jnp.int32),
            done=jnp.zeros(4, dtype=bool),
        )
        assert isinstance(batched, ann)

    def test_plain_scalar_hint_wrong_dtype_fails(self):
        @strux.struct(check=False)
        class Loss:
            loss: float

        batched = Loss(loss=jnp.zeros(4, dtype=jnp.int32))
        assert not isinstance(batched, Loss["batch"])

    def test_scalar_fields_no_trailing_space(self):
        # Float[Array, ""] is a scalar; batching should give "batch", not "batch "
        ann = Point["batch"]
        assert ann._field_hints["x"].dim_str == "batch"
        assert ann._field_hints["y"].dim_str == "batch"

    def test_empty_dims_is_noop(self):
        ann = Environment[""]
        assert ann._field_hints["hero_pos"].dim_str == "2"
        assert ann._field_hints["walls"].dim_str == "h w"

    def test_multi_dims(self):
        ann = Environment["batch time"]
        hints = ann._field_hints
        assert hints["hero_pos"].dim_str == "batch time 2"
        assert hints["walls"].dim_str == "batch time h w"

    def test_caching(self):
        a = Environment["batch"]
        b = Environment["batch"]
        assert a is b

    def test_different_dims_not_cached_together(self):
        a = Environment["batch"]
        b = Environment["time"]
        assert a is not b

    def test_annotation_name(self):
        ann = Environment["batch"]
        assert ann.__name__ == 'Environment["batch"]'


# # #
# Runtime isinstance checks


class TestInstanceCheck:
    def test_base_type_still_works(self):
        env = Environment(
            hero_pos=jnp.array([1, 2], dtype=jnp.int32),
            goal_pos=jnp.array([3, 4], dtype=jnp.int32),
            walls=jnp.zeros((5, 5), dtype=bool),
        )
        assert isinstance(env, Environment)

    def test_unbatched_fails_batched_annotation(self):
        env = Environment(
            hero_pos=jnp.array([1, 2], dtype=jnp.int32),
            goal_pos=jnp.array([3, 4], dtype=jnp.int32),
            walls=jnp.zeros((5, 5), dtype=bool),
        )
        assert not isinstance(env, Environment["batch"])

    def test_batched_passes(self):
        env = Environment(
            hero_pos=jnp.ones((3, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((3, 2), dtype=jnp.int32),
            walls=jnp.zeros((3, 5, 5), dtype=bool),
        )
        assert isinstance(env, Environment["batch"])

    def test_wrong_dtype_fails(self):
        # a wrong-dtype instance cannot be constructed under checking, so
        # use an unchecked class to exercise the isinstance path
        @strux.struct(check=False)
        class LooseEnv:
            hero_pos: Int[Array, "2"]

        env = LooseEnv(
            hero_pos=jnp.ones((3, 2), dtype=jnp.float32),  # wrong dtype
        )
        assert not isinstance(env, LooseEnv["batch"])

    def test_wrong_type_fails(self):
        assert not isinstance("not an env", Environment["batch"])
        assert not isinstance(42, Environment["batch"])

    def test_nested_struct_passes(self):
        world = World(
            env=Environment(
                hero_pos=jnp.ones((3, 2), dtype=jnp.int32),
                goal_pos=jnp.ones((3, 2), dtype=jnp.int32),
                walls=jnp.zeros((3, 5, 5), dtype=bool),
            ),
            score=jnp.array([1.0, 2.0, 3.0]),
        )
        assert isinstance(world, World["batch"])

    def test_nested_struct_fails_if_child_wrong(self):
        # an unbatched child with a batched sibling cannot be constructed
        # under checking, so use unchecked classes for the isinstance path
        @strux.struct(check=False)
        class LooseChild:
            pos: Int[Array, "2"]

        @strux.struct(check=False)
        class LooseWorld:
            env: LooseChild
            score: Float[Array, ""]

        world = LooseWorld(
            env=LooseChild(pos=jnp.ones((2,), dtype=jnp.int32)),  # unbatched
            score=jnp.array([1.0, 2.0, 3.0]),
        )
        assert not isinstance(world, LooseWorld["batch"])

    def test_meta_field_not_checked(self):
        @strux.struct(static_fieldnames=("name",))
        class WithMeta:
            pos: Int[Array, "2"]
            name: str
        obj = WithMeta(
            pos=jnp.ones((3, 2), dtype=jnp.int32),
            name="hello",
        )
        # check that meta field (name) is not checked during isinstance
        assert isinstance(obj, WithMeta["batch"])

    def test_scalar_struct_batched(self):
        point = Point(
            x=jnp.array([1.0, 2.0, 3.0]),
            y=jnp.array([4.0, 5.0, 6.0]),
        )
        assert isinstance(point, Point["batch"])


# # #
# Integration with jaxtyping + beartype runtime type checking


class TestJaxtypedIntegration:
    def test_correct_annotation_passes(self):
        @jaxtyped(typechecker=beartype)
        def step(env: Environment["batch"]) -> Environment["batch"]:
            return env

        env = Environment(
            hero_pos=jnp.ones((3, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((3, 2), dtype=jnp.int32),
            walls=jnp.zeros((3, 5, 5), dtype=bool),
        )
        result = step(env)
        assert isinstance(result, Environment)

    def test_wrong_annotation_raises(self):
        @jaxtyped(typechecker=beartype)
        def step(env: Environment["batch"]) -> Environment["batch"]:
            return env

        env = Environment(
            hero_pos=jnp.array([1, 2], dtype=jnp.int32),  # unbatched
            goal_pos=jnp.array([3, 4], dtype=jnp.int32),
            walls=jnp.zeros((5, 5), dtype=bool),
        )
        with pytest.raises(Exception):
            step(env)

    def test_nested_struct_jaxtyped(self):
        @jaxtyped(typechecker=beartype)
        def step(world: World["batch"]) -> World["batch"]:
            return world

        world = World(
            env=Environment(
                hero_pos=jnp.ones((3, 2), dtype=jnp.int32),
                goal_pos=jnp.ones((3, 2), dtype=jnp.int32),
                walls=jnp.zeros((3, 5, 5), dtype=bool),
            ),
            score=jnp.array([1.0, 2.0, 3.0]),
        )
        result = step(world)
        assert isinstance(result, World)

    def test_dimension_consistency_checked(self):
        """Within a @jaxtyped context, named dims must be consistent."""
        # such an instance cannot be constructed under checking, so use an
        # unchecked class to exercise the annotation path
        @strux.struct(check=False)
        class LooseEnv:
            hero_pos: Int[Array, "2"]
            goal_pos: Int[Array, "2"]

        @jaxtyped(typechecker=beartype)
        def step(env: LooseEnv["batch"]) -> LooseEnv["batch"]:
            return env

        # hero_pos batch=3, goal_pos batch=4 — inconsistent "batch" dim
        env = LooseEnv(
            hero_pos=jnp.ones((3, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((4, 2), dtype=jnp.int32),
        )
        with pytest.raises(Exception):
            step(env)
