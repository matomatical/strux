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

    def test_out_of_bounds_raises(self):
        batched = Point(x=jnp.arange(4.0), y=jnp.ones(4))
        with pytest.raises(IndexError, match="out of bounds"):
            batched[4]
        with pytest.raises(IndexError, match="out of bounds"):
            batched[-5]

    def test_negative_index(self):
        batched = Point(x=jnp.arange(4.0), y=jnp.ones(4))
        assert jnp.array_equal(batched[-1].x, batched[3].x)

    def test_unbatched_refuses_indexing(self):
        p = Point(x=jnp.float32(1.0), y=jnp.float32(2.0))
        with pytest.raises(TypeError, match="not batched"):
            p[0]

    def test_too_many_indices_raises(self):
        # a tuple longer than the batch rank would reach into element dims
        env = Environment(
            hero_pos=jnp.ones((4, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((4, 2), dtype=jnp.int32),
            walls=jnp.zeros((4, 5, 5), dtype=bool),
        )
        with pytest.raises(IndexError, match="too many indices"):
            env[0, 0]

    def test_tuple_indices_bounds_checked(self):
        env = Environment(
            hero_pos=jnp.ones((4, 3, 2), dtype=jnp.int32),
            goal_pos=jnp.ones((4, 3, 2), dtype=jnp.int32),
            walls=jnp.zeros((4, 3, 5, 5), dtype=bool),
        )
        with pytest.raises(IndexError, match="axis 1"):
            env[0, 3]

    def test_overlong_slice_clamps(self):
        # python slice semantics: slices clamp, they don't error
        batched = Point(x=jnp.arange(4.0), y=jnp.ones(4))
        assert batched[2:99].shape == (2,)


# # #
# Length and iteration


class TestLenIter:
    def test_len(self):
        batched = Point(x=jnp.arange(4.0), y=jnp.ones(4))
        assert len(batched) == 4

    def test_iteration_terminates_and_yields_elements(self):
        batched = Point(x=jnp.arange(3.0), y=jnp.ones(3))
        elements = list(batched)
        assert len(elements) == 3
        assert all(isinstance(p, Point) for p in elements)
        assert jnp.array_equal(elements[2].x, jnp.float32(2.0))

    def test_unbatched_refuses_len_and_iteration(self):
        p = Point(x=jnp.float32(1.0), y=jnp.float32(2.0))
        with pytest.raises(TypeError, match="not batched"):
            len(p)
        with pytest.raises(TypeError, match="not batched"):
            next(iter(p))

    def test_multi_batch_iterates_leading_dim(self):
        batched = Point(x=jnp.ones((4, 3)), y=jnp.ones((4, 3)))
        elements = list(batched)
        assert len(elements) == 4
        assert elements[0].shape == (3,)


# # #
# Solved-batch caching


class TestSolutionCache:
    def test_shape_cached_at_construction(self):
        batched = Point(x=jnp.arange(4.0), y=jnp.ones(4))
        assert batched.shape == (4,)
        # validation stored the solution: no field access needed to answer
        assert batched.__dict__["_strux_candidates"] == frozenset({(4,)})

    def test_unflattened_instances_solve_lazily(self):
        import jax
        batched = Point(x=jnp.arange(4.0), y=jnp.ones(4))
        doubled = jax.tree.map(lambda a: a * 2, batched)   # bypasses init
        assert "_strux_candidates" not in doubled.__dict__
        assert doubled.shape == (4,)
        assert "_strux_candidates" in doubled.__dict__

    def test_nested_construction_uses_child_cache(self):
        world = World(
            env=Environment(
                hero_pos=jnp.ones((4, 2), dtype=jnp.int32),
                goal_pos=jnp.ones((4, 2), dtype=jnp.int32),
                walls=jnp.zeros((4, 5, 5), dtype=bool),
            ),
            score=jnp.array([1.0, 2.0, 3.0, 4.0]),
        )
        assert world.shape == (4,)
        assert world.env.shape == (4,)


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

    def test_construction_enforces_names(self):
        # a name takes one consistent size across the class's fields,
        # checked at construction
        @strux.struct
        class Affine:
            weights: Float[Array, "n_in n_out"]
            biases: Float[Array, "n_out"]

        Affine(weights=jnp.ones((4, 8)), biases=jnp.zeros(8))
        with pytest.raises(strux.ValidationError, match="inconsistent dim 'n_out'"):
            Affine(weights=jnp.ones((4, 8)), biases=jnp.zeros(3))

    def test_batched_construction_binds_element_dims(self):
        # names are trailing element dims: leading batch dims don't
        # interfere with the binding
        @strux.struct
        class Affine:
            weights: Float[Array, "n_in n_out"]
            biases: Float[Array, "n_out"]

        Affine(weights=jnp.ones((10, 4, 8)), biases=jnp.zeros((10, 8)))
        with pytest.raises(strux.ValidationError, match="inconsistent dim"):
            Affine(weights=jnp.ones((10, 4, 8)), biases=jnp.zeros((10, 3)))

    def test_anonymous_dims_do_not_bind(self):
        # "_" is jaxtyping's anonymous dim: it never binds, so ragged
        # sizes across fields (or container elements) are expressible
        @strux.struct
        class Ragged:
            a: Float[Array, "_"]
            b: Float[Array, "_"]

        Ragged(a=jnp.ones(3), b=jnp.ones(5))

    def test_batched_instances_bind_element_dims(self):
        e = Environment(
            hero_pos=jnp.zeros((7, 2), jnp.int32),
            goal_pos=jnp.zeros((7, 2), jnp.int32),
            walls=jnp.zeros((7, 5, 6), bool),
        )
        assert strux.tree_dims(e) == {"h": 5, "w": 6}

    def test_names_scoped_per_class(self):
        # a name's scope is the class whose annotations mention it: nested
        # structs bind their own names at their own construction, so two
        # Layer fields (or elements) may take different sizes — the MLP
        # case (layer widths differ) constructs fine
        @strux.struct
        class Layer:
            w: Float[Array, "n n"]

        @strux.struct
        class Stack:
            layers: tuple[Layer, ...]

        stack = Stack(
            layers=(Layer(w=jnp.ones((3, 3))), Layer(w=jnp.zeros((4, 4)))),
        )
        assert strux.tree_dims(stack) == {}     # Stack mentions no names
        assert strux.tree_dims(stack.layers[1]) == {"n": 4}
        # within one Layer, its own names still bind
        with pytest.raises(strux.ValidationError, match="inconsistent dim 'n'"):
            Layer(w=jnp.ones((3, 4)))

    def test_container_elements_share_the_class_namespace(self):
        # container elements are part of this class's annotation: a named
        # dim unifies across elements and with sibling fields (use "_"
        # for ragged elements)
        @strux.struct
        class Bank:
            layers: tuple[Float[Array, "n"], ...]
            bias: Float[Array, "n"]

        Bank(layers=(jnp.ones(3), jnp.zeros(3)), bias=jnp.ones(3))
        with pytest.raises(strux.ValidationError, match="inconsistent dim 'n'"):
            Bank(layers=(jnp.ones(3), jnp.zeros(5)), bias=jnp.ones(3))
