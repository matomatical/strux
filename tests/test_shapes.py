"""
Tests for shapes and indexing: batch shape, batch indexing, and symbolic
dim binding (tree_dims).
"""

import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

import strux

from example_structs import Environment, Point, World


# # #
# Batch shape


class TestShape:
    def test_unbatched(self):
        env = Environment(
            hero_pos=jnp.array([1, 2], dtype=jnp.int32),
            goal_pos=jnp.array([3, 4], dtype=jnp.int32),
            walls=jnp.zeros((5, 5), dtype=bool),
        )
        assert env.shape == ()

    def test_single_batch_dim(self):
        env = Environment(
            hero_pos=jnp.ones((4, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((4, 2), dtype=jnp.int32),
            walls=jnp.zeros((4, 5, 5), dtype=bool),
        )
        assert env.shape == (4,)

    def test_multi_batch_dims(self):
        env = Environment(
            hero_pos=jnp.ones((4, 3, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((4, 3, 2), dtype=jnp.int32),
            walls=jnp.zeros((4, 3, 5, 5), dtype=bool),
        )
        assert env.shape == (4, 3)

    def test_scalar_fields(self):
        p = Point(x=jnp.array([1.0, 2.0]), y=jnp.array([3.0, 4.0]))
        assert p.shape == (2,)

    def test_plain_scalar_hint_fields(self):
        @strux.struct
        class Metrics:
            loss: float
            step: int

        # python scalars have batch shape ()
        assert Metrics(loss=1.0, step=7).shape == ()
        # a batched instance: all of the value's dims are batch dims
        batched = Metrics(loss=jnp.zeros(4), step=jnp.arange(4))
        assert batched.shape == (4,)

    def test_nested_struct(self):
        world = World(
            env=Environment(
                hero_pos=jnp.ones((4, 2), dtype=jnp.int32),
                goal_pos=jnp.ones((4, 2), dtype=jnp.int32),
                walls=jnp.zeros((4, 5, 5), dtype=bool),
            ),
            score=jnp.array([1.0, 2.0, 3.0, 4.0]),
        )
        assert world.shape == (4,)

    def test_inconsistent_batch_raises(self):
        # checked construction refuses inconsistent batch shapes outright
        with pytest.raises(strux.ValidationError, match="inconsistent batch"):
            Environment(
                hero_pos=jnp.ones((3, 2), dtype=jnp.int32),
                goal_pos=jnp.ones((4, 2), dtype=jnp.int32),
                walls=jnp.zeros((3, 5, 5), dtype=bool),
            )
        # .shape raises the same for an unchecked inconsistent struct
        @strux.struct(check=False)
        class LoosePair:
            u: Float[Array, ""]
            v: Float[Array, ""]

        pair = LoosePair(u=jnp.ones(3), v=jnp.ones(4))
        with pytest.raises(strux.ValidationError, match="inconsistent batch"):
            pair.shape

    def test_shape_field_collision_warns(self):
        with pytest.warns(UserWarning, match="field named 'shape'"):
            @strux.struct
            class HasShape:
                shape: int
                x: float


# # #
# Indexing


class TestGetitem:
    def test_integer_index(self):
        env = Environment(
            hero_pos=jnp.ones((4, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((4, 2), dtype=jnp.int32),
            walls=jnp.zeros((4, 5, 5), dtype=bool),
        )
        e = env[0]
        assert isinstance(e, Environment)
        assert e.hero_pos.shape == (2,)
        assert e.walls.shape == (5, 5)

    def test_slice(self):
        env = Environment(
            hero_pos=jnp.ones((4, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((4, 2), dtype=jnp.int32),
            walls=jnp.zeros((4, 5, 5), dtype=bool),
        )
        e = env[1:3]
        assert isinstance(e, Environment)
        assert e.hero_pos.shape == (2, 2)
        assert e.walls.shape == (2, 5, 5)

    def test_multi_batch_index(self):
        env = Environment(
            hero_pos=jnp.ones((4, 3, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((4, 3, 2), dtype=jnp.int32),
            walls=jnp.zeros((4, 3, 5, 5), dtype=bool),
        )
        e = env[0]
        assert e.hero_pos.shape == (3, 2)
        assert e.walls.shape == (3, 5, 5)
        e2 = env[0, 1]
        assert e2.hero_pos.shape == (2,)
        assert e2.walls.shape == (5, 5)

    def test_advanced_indexing(self):
        env = Environment(
            hero_pos=jnp.ones((4, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((4, 2), dtype=jnp.int32),
            walls=jnp.zeros((4, 5, 5), dtype=bool),
        )
        e = env[jnp.array([0, 2])]
        assert e.hero_pos.shape == (2, 2)
        assert e.walls.shape == (2, 5, 5)

    def test_nested_struct_index(self):
        world = World(
            env=Environment(
                hero_pos=jnp.ones((4, 2), dtype=jnp.int32),
                goal_pos=jnp.ones((4, 2), dtype=jnp.int32),
                walls=jnp.zeros((4, 5, 5), dtype=bool),
            ),
            score=jnp.array([1.0, 2.0, 3.0, 4.0]),
        )
        w = world[0]
        assert isinstance(w, World)
        assert isinstance(w.env, Environment)
        assert w.env.hero_pos.shape == (2,)
        assert w.score.shape == ()

    def test_index_preserves_values(self):
        env = Environment(
            hero_pos=jnp.array([[10, 20], [30, 40]], dtype=jnp.int32),
            goal_pos=jnp.array([[50, 60], [70, 80]], dtype=jnp.int32),
            walls=jnp.zeros((2, 5, 5), dtype=bool),
        )
        e = env[1]
        assert jnp.array_equal(e.hero_pos, jnp.array([30, 40]))
        assert jnp.array_equal(e.goal_pos, jnp.array([70, 80]))


# # #
# Symbolic dim binding (tree_dims)


class TestTreeDims:
    def test_binds_names_across_fields(self):
        @strux.struct
        class Affine:
            weights: Float[Array, "n_in n_out"]
            biases: Float[Array, "n_out"]

        net = Affine(weights=jnp.ones((4, 8)), biases=jnp.zeros(8))
        assert strux.tree_dims(net) == {"n_in": 4, "n_out": 8}

    def test_inconsistent_names_raise(self):
        @strux.struct(check=False)
        class Affine:
            weights: Float[Array, "n_in n_out"]
            biases: Float[Array, "n_out"]

        broken = Affine(weights=jnp.ones((4, 8)), biases=jnp.zeros(3))
        with pytest.raises(strux.ValidationError, match="inconsistent dim 'n_out'"):
            strux.tree_dims(broken)

    def test_construction_does_not_enforce_names(self):
        # documented v1 semantics: symbolic dims are rank-only at
        # construction; tree_dims is the stricter (on-demand) check
        @strux.struct
        class Affine:
            weights: Float[Array, "n_in n_out"]
            biases: Float[Array, "n_out"]

        Affine(weights=jnp.ones((4, 8)), biases=jnp.zeros(3))    # accepted

    def test_batched_instances_bind_element_dims(self):
        e = Environment(
            hero_pos=jnp.zeros((7, 2), jnp.int32),
            goal_pos=jnp.zeros((7, 2), jnp.int32),
            walls=jnp.zeros((7, 5, 6), bool),
        )
        assert strux.tree_dims(e) == {"h": 5, "w": 6}

    def test_nested_and_container_binding(self):
        @strux.struct
        class Layer:
            w: Float[Array, "n n"]

        @strux.struct
        class Stack:
            layers: tuple[Layer, ...]

        stack = Stack(layers=(Layer(w=jnp.ones((3, 3))), Layer(w=jnp.zeros((3, 3)))))
        assert strux.tree_dims(stack) == {"n": 3}
        broken = Stack(layers=(Layer(w=jnp.ones((3, 3))), Layer(w=jnp.zeros((4, 4)))))
        with pytest.raises(strux.ValidationError, match="inconsistent dim 'n'"):
            strux.tree_dims(broken)
