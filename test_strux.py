import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Bool, Float, Int, jaxtyped
from beartype import beartype

import strux


# --- Test structs ---

@strux.struct
class Point:
    x: Float[Array, ""]
    y: Float[Array, ""]


@strux.struct
class Environment:
    hero_pos: Int[Array, "2"]
    goal_pos: Int[Array, "2"]
    walls: Bool[Array, "h w"]


@strux.struct
class World:
    env: Environment
    score: Float[Array, ""]


@lambda cls: strux.struct(cls, static_fieldnames=("name",))
class WithMeta:
    pos: Int[Array, "2"]
    name: str


# --- Annotation expansion (static behaviour) ---

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
        ann = WithMeta["batch"]
        hints = ann._field_hints
        # jaxtyping data field is expanded
        assert hints["pos"].dim_str == "batch 2"
        # meta field is not included
        assert "name" not in hints

    def test_non_batchable_data_field_raises(self):
        @strux.struct
        class Bad:
            pos: Int[Array, "2"]
            name: str

        with pytest.raises(TypeError, match="Cannot batch data field 'name'"):
            Bad["batch"]

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


# --- Runtime isinstance checks ---

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
        env = Environment(
            hero_pos=jnp.ones((3, 2), dtype=jnp.float32),  # wrong dtype
            goal_pos=jnp.ones((3, 2), dtype=jnp.int32),
            walls=jnp.zeros((3, 5, 5), dtype=bool),
        )
        assert not isinstance(env, Environment["batch"])

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
        world = World(
            env=Environment(
                hero_pos=jnp.ones((2,), dtype=jnp.int32),  # unbatched
                goal_pos=jnp.ones((2,), dtype=jnp.int32),
                walls=jnp.zeros((5, 5), dtype=bool),
            ),
            score=jnp.array([1.0, 2.0, 3.0]),
        )
        assert not isinstance(world, World["batch"])

    def test_meta_field_not_checked(self):
        # meta field (name) is not checked during isinstance
        obj = WithMeta(
            pos=jnp.ones((3, 2), dtype=jnp.int32),
            name="hello",
        )
        assert isinstance(obj, WithMeta["batch"])

    def test_scalar_struct_batched(self):
        point = Point(
            x=jnp.array([1.0, 2.0, 3.0]),
            y=jnp.array([4.0, 5.0, 6.0]),
        )
        assert isinstance(point, Point["batch"])


# --- Integration with jaxtyping + beartype runtime type checking ---

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
        @jaxtyped(typechecker=beartype)
        def step(env: Environment["batch"]) -> Environment["batch"]:
            return env

        # hero_pos batch=3, goal_pos batch=4 — inconsistent "batch" dim
        env = Environment(
            hero_pos=jnp.ones((3, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((4, 2), dtype=jnp.int32),
            walls=jnp.zeros((3, 5, 5), dtype=bool),
        )
        with pytest.raises(Exception):
            step(env)
